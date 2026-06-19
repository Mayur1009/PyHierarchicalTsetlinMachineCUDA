# Copyright (c) 2026 Ole-Christoffer Granmo and the University of Agder

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import deque
from random import randint
import cupy as cp
import numpy as np

import PyHierarchicalTsetlinMachineCUDA.kernels as kernels

OR_GROUP = " ∨* "
OR_ALTERNATIVES = " ∨ "
AND_GROUP = " ∧ "

VANILLA_TM = 0
WEIGHTED_TM = 1
COALESCED_TM = 2


class CommonTsetlinMachine():

	def __init__(self, number_of_clauses, T, s, hierarchy_structure, q=1.0, log_scale=False, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1), seed=None):
		# Params
		self.number_of_clauses = number_of_clauses
		self.number_of_state_bits = number_of_state_bits
		self.T = T
		self.s = s
		self.q = q
		self.log_scale = log_scale
		self.hierarchy_structure = hierarchy_structure
		self.depth = len(hierarchy_structure)

		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.append_negated = append_negated
		self.grid = grid
		self.block = block

        # Make sure child classes set these
		if not hasattr(self, 'negative_clauses'):
			self.negative_clauses = 1
		if not hasattr(self, 'flip_polarity'):
			self.flip_polarity = 0
		if not hasattr(self, 'tm_type'):
			self.tm_type = VANILLA_TM

		# Calculates the number of nodes at each level of the hierarchy
		self.hierarchy_size = [0] * (self.depth + 1)
		self.hierarchy_size[self.depth] = 1
		for d in range(self.depth - 1):
			self.hierarchy_size[self.depth - d - 1] = self.hierarchy_structure[self.depth - d - 1][1] * self.hierarchy_size[self.depth - d]

		# Represents hierarchy structure for transfer to GPU
		self.hierarchy_structure_factors = [0] * (self.depth - 1)
		self.hierarchy_structure_type = [0] * (self.depth - 1)
		for d in range(1, self.depth):
			self.hierarchy_structure_factors[d-1] = self.hierarchy_structure[d][1]
			if self.hierarchy_structure[d][0] == OR_ALTERNATIVES:
				self.hierarchy_structure_type[d-1] = 1
			elif self.hierarchy_structure[d][0] == OR_GROUP:
				self.hierarchy_structure_type[d-1] = 2

		# Calculates total number of features spanned by the hierarchy
		self.number_of_features_hierarchy = 1
		for d in range(self.depth - 1, -1, -1):
			if (self.hierarchy_structure[d][0] == OR_GROUP or self.hierarchy_structure[d][0] == AND_GROUP):
				self.number_of_features_hierarchy *= self.hierarchy_structure[d][1]

		# Calculates literal chunks per leaf
		self.number_of_features_per_leaf = self.hierarchy_structure[0][1]
		if self.append_negated:
			self.number_of_literals = self.number_of_features_hierarchy * 2
			self.number_of_literals_per_leaf = self.number_of_features_per_leaf * 2
			self.number_of_literal_chunks_per_leaf = int((self.number_of_literals_per_leaf - 1) / 32 + 1)
		else:
			self.number_of_literals = self.number_of_features_hierarchy
			self.number_of_literals_per_leaf = self.number_of_features_per_leaf
			self.number_of_literal_chunks_per_leaf = int((self.number_of_literals_per_leaf - 1) / 32 + 1)

		# Calculates the number of literal chunks for the full hierarchy
		self.hierarchy_size[0] = self.number_of_literal_chunks_per_leaf * self.hierarchy_size[1]

		# Calculates number of literal chunks overall for the feature vector (ignores OR alternatives)
		self.number_of_literal_chunks = self.number_of_literal_chunks_per_leaf
		for d in range(self.depth - 1, 0, -1):
			if (self.hierarchy_structure[d][0] == OR_GROUP or self.hierarchy_structure[d][0] == AND_GROUP):
				self.number_of_literal_chunks *= self.hierarchy_structure[d][1]

		self.seed = randint(1, 2**30) if seed is None else (seed if seed > 0 else seed + 1)

		self.cuda_modules()

		self.first = True


	def cuda_modules(self):
		parameters = f"""
#define CLAUSES {self.number_of_clauses}
#define DEPTH {self.depth}
#define COMPONENTS {self.hierarchy_size[1]}
#define LITERALS_PER_LEAF {self.number_of_literals_per_leaf}
#define TA_CHUNKS_PER_LEAF {self.number_of_literal_chunks_per_leaf}
#define LITERAL_CHUNKS {self.number_of_literal_chunks}
#define STATE_BITS {self.number_of_state_bits}
#define BOOST_TRUE_POSITIVE_FEEDBACK {self.boost_true_positive_feedback}
#define S {float(self.s)}f
#define THRESHOLD {self.T}
#define Q {float(self.q)}f
#define LOG_SCALE {int(self.log_scale)}

#define NEGATIVE_CLAUSES {self.negative_clauses}
#define FLIP_POLARITY {self.flip_polarity}

#define VANILLA_TM {VANILLA_TM}
#define WEIGHTED_TM {WEIGHTED_TM}
#define COALESCED_TM {COALESCED_TM}
		"""
		
		mod_prepare = cp.RawModule(code=parameters + kernels.code_header + kernels.code_prepare)
		self.prepare_weights = mod_prepare.get_function("prepare_weights")
		self.prepare_hierarchy = mod_prepare.get_function("prepare_hierarchy")

		mod_update = cp.RawModule(code=parameters + kernels.code_header + kernels.code_update)
		self.update_hierarchy = mod_update.get_function("update_hierarchy")
		self.update_weights = mod_update.get_function("update_weights")
		self.evaluate_leaves = mod_update.get_function("evaluate_leaves")
		self.max_clause_output = mod_update.get_function("max_clause_output")
		self.evaluate_final = mod_update.get_function("evaluate_final")
		self.rescale_final = mod_update.get_function("rescale_final")
		self.evaluate_and_groups = mod_update.get_function("evaluate_and_groups")
		self.propagate_and_group_false_truth_values = mod_update.get_function("propagate_and_group_false_truth_values")
		self.propagate_or_group_false_truth_values = mod_update.get_function("propagate_or_group_false_truth_values")
		self.evaluate_or_groups = mod_update.get_function("evaluate_or_groups")
		self.evaluate_or_alternatives = mod_update.get_function("evaluate_or_alternatives")

		# CUDA modules for encoding input data
		mod_encode = cp.RawModule(code=kernels.code_encode)
		self.prepare_encode_hierarchy = mod_encode.get_function("prepare_encode_hierarchy")
		self.encode_hierarchy = mod_encode.get_function("encode_hierarchy")

		mod_clauses = cp.RawModule(code=parameters + kernels.code_clauses)
		self.kernel_get_ta_states = mod_clauses.get_function("get_ta_states")

	def encode_X(self, X):
		number_of_examples = X.shape[0]

		# Allocates GPU memory for input data
		X_gpu = cp.asarray(X, dtype=np.uint32)
		encoded_X_hierarchy_gpu = cp.zeros((number_of_examples, self.number_of_literal_chunks), dtype=cp.uint32)

		# Prepares for leaf encoding of the input data
		self.prepare_encode_hierarchy(
			self.grid, self.block,
			(X_gpu, encoded_X_hierarchy_gpu, np.int32(self.number_of_literal_chunks), np.int32(number_of_examples))
		)

		# Encodes the input data split across the leaves
		self.encode_hierarchy(
			self.grid, self.block,
			(X_gpu, encoded_X_hierarchy_gpu, np.int32(self.number_of_features_hierarchy), np.int32(self.number_of_literal_chunks), np.int32(self.hierarchy_size[1]), np.int32(self.number_of_features_per_leaf), np.int32(self.number_of_literal_chunks_per_leaf), np.int32(self.append_negated), np.int32(number_of_examples))
		)

		return encoded_X_hierarchy_gpu

	def allocate_gpu_memory(self):
        # Votes per level
		self.hierarchy_votes = [cp.zeros((self.number_of_clauses, self.hierarchy_size[d]), dtype=cp.float32) for d in range(1, self.depth)] + [cp.zeros((self.number_of_clauses, 1), dtype=cp.float32)]

		# GPU memory for storing hierarchy structure
		self.hierarchy_structure_factors_gpu = cp.asarray(self.hierarchy_structure_factors, dtype=cp.int32)

		# GPU memory for storing hierarchy structure
		self.hierarchy_structure_type_gpu = cp.asarray(self.hierarchy_structure_type, dtype=cp.int32)

		# GPU memory for storing Tsetlin Automata states
		self.ta_state_hierarchy_gpu = cp.zeros((
		    self.number_of_clauses,
		    self.hierarchy_size[1],
		    self.number_of_literal_chunks_per_leaf,
		    self.number_of_state_bits
		), dtype=cp.uint32)
		self.clause_weights_gpu = cp.zeros((self.number_of_outputs, self.number_of_clauses), dtype=cp.int32)
		self.component_weights_gpu = cp.zeros((self.number_of_clauses, self.hierarchy_size[1]), dtype=cp.int32) # Only positive weights...

	def ta_action(self, clause: int, leaf: int, ta: int) -> bool:
		"""Get the include/exclude action of a TA, indexed by (clause, leaf, ta)"""
		return self.ta_state_hierarchy_gpu[clause, leaf, ta // 32, self.number_of_state_bits - 1].get() & (1 << (ta % 32)) > 0

	def ta_state(self, clause: int, leaf: int, ta: int) -> int:
		"""Get the state of a TA, indexed by (clause, leaf, ta)"""
		bits = self.ta_state_hierarchy_gpu[clause, leaf, ta // 32, :].get()
		ta_bit_active = (bits >> (ta % 32)) & 1
		bit_values = 1 << np.arange(self.number_of_state_bits, dtype=np.uint32)
		return int(np.dot(ta_bit_active, bit_values))

	# Transform input data for processing at next layer
	def transform(self, X):
		None # To be updated

	def initialize_weights_and_ta_states(self):
		class_sum_gpu = cp.zeros(self.number_of_outputs, dtype=cp.float32)
		self.prepare_weights(self.grid, self.block, (
			np.uint64(self.seed),
			np.int32(self.tm_type),
			np.int32(self.number_of_outputs),
			self.clause_weights_gpu,
			class_sum_gpu
		))

		self.prepare_hierarchy(self.grid, self.block, (
			np.uint64(self.seed),
			np.int32(self.number_of_outputs),
			self.ta_state_hierarchy_gpu,
			self.clause_weights_gpu,
			class_sum_gpu
		))

	def evaluate_hierarchy(self, encoded_X_hierarchy, e):
		# Initializes class sums to zero
		class_sum_gpu = cp.zeros(self.number_of_outputs, dtype=cp.float32)

		# Evaluates all the hierarchy leaves in parallel
		self.evaluate_leaves(self.grid, self.block, (
			self.ta_state_hierarchy_gpu,
			self.component_weights_gpu,
			self.hierarchy_votes[0],
			np.int32(self.depth),
			self.hierarchy_structure_factors_gpu,
			self.hierarchy_structure_type_gpu,
			encoded_X_hierarchy,
			np.int32(e)
		))

		# Propagates votes bottom-up in the hierarchy, starting from the clause components (leaves)
		for d in range(1, self.depth):
			if (self.hierarchy_structure[d][0] == AND_GROUP):
				self.evaluate_and_groups(self.grid, self.block, (
					self.hierarchy_votes[d-1],
					self.hierarchy_votes[d],
					np.int32(self.hierarchy_size[d + 1]),
					np.int32(self.hierarchy_structure[d][1])
				))
			elif self.hierarchy_structure[d][0] == OR_GROUP:
				self.evaluate_or_groups(self.grid, self.block, (
					self.hierarchy_votes[d-1],
					self.hierarchy_votes[d],
					np.int32(self.hierarchy_size[d + 1]),
					np.int32(self.hierarchy_structure[d][1])
				))
			elif self.hierarchy_structure[d][0] == OR_ALTERNATIVES:
				self.evaluate_or_alternatives(self.grid, self.block, (
					self.hierarchy_votes[d-1],
					self.hierarchy_votes[d],
					np.int32(self.hierarchy_size[d + 1]),
					np.int32(self.hierarchy_structure[d][1])
				))
			else:
				raise ValueError("Unknown Node Type!")

		# self.clause_output_max[:] = np.finfo(np.float32).min
		# cuda.memcpy_htod(self.clause_output_max_gpu, self.clause_output_max)

		# if self.log_scale:
		# 	self.max_clause_output.prepared_call(
		# 		self.grid,
		# 		self.block,
		# 		np.int32(self.number_of_outputs),
		# 		self.hierarchy_votes[self.depth-1],
		# 		self.clause_output_max_gpu
		# 	)
		# 	cuda.Context.synchronize()

		# Adds up the votes from each clause (hierarchy root)
		self.evaluate_final(self.grid, self.block, (
			np.int32(self.number_of_outputs),
			self.hierarchy_votes[self.depth-1],
			self.clause_weights_gpu,
			class_sum_gpu
		))

		return class_sum_gpu

		# if self.log_scale:
		# 	self.rescale_final.prepared_call(
		# 		self.grid,
		# 		self.block,
		# 		np.int32(self.number_of_outputs),
		# 		self.clause_output_max_gpu,
		# 		self.class_sum_gpu
		# 	)
		# 	cuda.Context.synchronize()

	def _fit(self, X, encoded_Y, epochs=100, incremental=False):
		if self.number_of_features_hierarchy != X.shape[1]:
			raise ValueError("The number of features spanned by hierarchy does not align with the input data.")

		number_of_examples = X.shape[0]

		if self.first:
			# Allocates memory and prepares weights and Tsetlin automata states on first run 
			self.allocate_gpu_memory()

			self.initialize_weights_and_ta_states()

			self.first = False
		elif not incremental:
			# Re-initializes weights and Tsetlin automata states if training is not incremental
			self.initialize_weights_and_ta_states()

		# Allocates GPU memory for training data
		Y_gpu = cp.asarray(encoded_Y, dtype=cp.uint32)
		encoded_X_hierarchy_training_gpu = self.encode_X(X)

		for epoch in range(epochs):
			for e in range(number_of_examples):
				class_sum_gpu = self.evaluate_hierarchy(encoded_X_hierarchy_training_gpu, e)


				# Propagates the root value and any intermittent node values back to the leaves.
				# The purpose is to determine which leaves only has True nodes on the path from leaf to root.
				for d in range(self.depth-1, 0, -1):
					if self.hierarchy_structure[d][0] != OR_GROUP:
						self.propagate_and_group_false_truth_values(self.grid, self.block, (
							self.hierarchy_votes[d-1],
							self.hierarchy_votes[d],
							np.int32(self.hierarchy_size[d + 1]),
							np.int32(self.hierarchy_structure[d][1])
						))
					else:
						self.propagate_or_group_false_truth_values(self.grid, self.block, (
							np.uint64(self.seed),
							self.hierarchy_votes[d-1],
							self.hierarchy_votes[d],
							np.int32(self.hierarchy_size[d + 1]),
							np.int32(self.hierarchy_structure[d][1]),
							np.int32(e)
						))

				# Updates the clause components (leaves) based on the propagated truth values
				self.update_hierarchy(self.grid, self.block, (
					np.uint64(self.seed),
					np.int32(self.number_of_outputs),
					self.ta_state_hierarchy_gpu,
					self.clause_weights_gpu,
					self.hierarchy_votes[0],
					np.int32(self.depth),
					self.hierarchy_structure_factors_gpu,
					self.hierarchy_structure_type_gpu,
					class_sum_gpu,
					encoded_X_hierarchy_training_gpu,
					Y_gpu,
					np.int32(e)
				))

				# Updates the clause weights
				if (self.tm_type in [WEIGHTED_TM, COALESCED_TM]):
					self.update_weights(self.grid, self.block, (
						np.uint64(self.seed),
						np.int32(self.tm_type),
						np.int32(self.number_of_outputs),
						self.clause_weights_gpu,
						self.hierarchy_votes[self.depth-1],
						class_sum_gpu,
						Y_gpu,
						np.int32(e)
					))
		return
       
	def _score(self, X, clip=True):
		number_of_examples = X.shape[0]
		encoded_X_hierarchy_test_gpu = self.encode_X(X)
		class_sum = cp.zeros((self.number_of_outputs, number_of_examples), dtype=cp.float32)
		for e in range(number_of_examples):
			class_sum[:, e] = self.evaluate_hierarchy(encoded_X_hierarchy_test_gpu, e)

		if clip:
			return np.clip(class_sum.get(), -self.T, self.T)
		else:
			return class_sum.get()

	def get_ta_states(self) -> np.ndarray:
		"""
		Get state value for each TA.
		Returns: Numpy array of shape (number_of_clauses, number_of_clause_components, number_of_literals_per_leaf)
		"""
		# Mem Allocation
		ta_states_gpu = cp.zeros((self.number_of_clauses, self.hierarchy_size[1], self.number_of_literals_per_leaf), dtype=cp.uint32)

		# Calculate grid size based on the kernel
		total = self.number_of_clauses * self.hierarchy_size[1] * self.number_of_literals_per_leaf
		grid = (((total + self.block[0] - 1) // self.block[0]), 1, 1)
		self.kernel_get_ta_states(grid, self.block, (self.ta_state_hierarchy_gpu, ta_states_gpu))

		# Copy back to CPU
		return ta_states_gpu.get()

	def get_literals(self):
		"""
		Get included literals for each clause.
		Returns: Numpy array of shape (number_of_clauses, number_of_clause_components, number_of_literals_per_leaf)
		"""
		return (self.get_ta_states() >= (1 << (self.number_of_state_bits - 1))).astype(np.uint8)

	def map_ta_id_to_feature_id(self):
		"""
		Return an array of shape(number_of_clause_components, number_of_literals_per_leaf). That is the total number of TAs in a clause. Maps each TA id to a feature_id in the input. In each component, the first half of the TAs correspond to the positive features, and the second half correspond to the negated features.
		"""
		# BFS top-down traversal
		q = deque()
		q.append((self.depth - 1, 0, 0)) # (level, node_id, group_id)

		comp_grps = -1 * np.ones(self.hierarchy_size[1], dtype=np.int32)
		while q:
			level, node_id, group_id = q.popleft()

			if level == 0:
				# This is the leaf component
				comp_grps[node_id] = group_id
				continue

			n_children = self.hierarchy_structure[level][1]
			is_alt = (self.hierarchy_structure[level][0] == OR_ALTERNATIVES)
			for child_pos in range(n_children):
				child_id = node_id * n_children + child_pos
				if is_alt:
					# All children share the same features
					child_group_id = group_id
				else:
					# Features are partitioned among the children
					child_group_id = group_id * n_children + child_pos

				q.append((level - 1, child_id, child_group_id))

		# map each TA in a component to a feature
		half = self.number_of_literals_per_leaf // 2
		lit_ids = np.arange(self.number_of_literals_per_leaf)
		local_feats = lit_ids % half if self.append_negated else lit_ids
		fmap = comp_grps[:, None] * (half if self.append_negated else self.number_of_literals_per_leaf) + local_feats[None, :]

		return fmap

	def calc_hierarchy_votes(self, X, clip=True):
		"""
		Get the clause activation information for each sample in X.
		"""
		assert not self.first, "Model must be trained before getting activations."

		number_of_examples = X.shape[0]
		encoded_X_hierarchy_test_gpu = self.encode_X(X)

		class_sum = np.zeros((self.number_of_outputs, number_of_examples), dtype=cp.float32)
		hierarchy_votes = []
		for e in range(number_of_examples):
			class_sum[:, e] = self.evaluate_hierarchy(encoded_X_hierarchy_test_gpu, e).get()

			hierarchy_votes_example = []
			for d in range(self.depth):
				hierarchy_votes_example.append(self.hierarchy_votes[d].get().reshape((self.number_of_clauses, int(self.hierarchy_size[d+1]))))

			hierarchy_votes.append(hierarchy_votes_example)

		return hierarchy_votes, np.clip(class_sum, -self.T, self.T) if clip else class_sum

	def print_hierarchy(self, print_ta_state=False):
		for i in range(self.number_of_clauses):
			print("\nCLAUSE #%d: " % (i), end='')

			previous_index = np.ones((self.depth-1), dtype=np.int32)*-1
			for j in range(self.hierarchy_size[1]):
				component_remainder = j
				size = 1

				left = []
				right = []
				inside = []
				feature_base = 0
				size = self.hierarchy_structure[0][1]
				for d in range(1, self.depth):
					depth_d_node_index = component_remainder % self.hierarchy_structure[d][1]
					component_remainder = component_remainder // self.hierarchy_structure[d][1]

					if self.hierarchy_structure[d][0] != OR_ALTERNATIVES:
						feature_base += size * depth_d_node_index 
						size *= self.hierarchy_structure[d][1];

					if previous_index[d-1] == -1:
						left.append("(")
					elif depth_d_node_index == 0 and previous_index[d-1] != depth_d_node_index:
						right.append(")")
						left.insert(0, "(")
					elif previous_index[d-1] != depth_d_node_index:
						inside.append(self.hierarchy_structure[d][0])
					
					previous_index[d-1] = depth_d_node_index

				for s in right:
					print(s, end='')

				for s in inside:
					print(s, end='')

				for s in left:
					print(s, end='')

				l = []
				for k in range(self.number_of_literals_per_leaf):
					if self.ta_action(i, j, k):
						if k < self.number_of_literals_per_leaf // 2:
							if print_ta_state:
								l.append("x%d(%d)" % (feature_base + k, self.ta_state(i, j, k)))
							else:
								l.append("x%d" % (feature_base + k,))
						else:
							if print_ta_state:
								l.append("¬x%d(%d)" % (feature_base + k - self.number_of_literals_per_leaf // 2, self.ta_state(i, j, k)))
							else:
								l.append("¬x%d" % (feature_base + k - self.number_of_literals_per_leaf // 2,))
				
				if len(l) > 1:
					print("(" + " ∧ ".join(l) + ")", end = '')
				elif len(l) == 1:
					print(l[0], end = '')

			print(")" * (self.depth - 1))

	def save(self) -> dict:
		return {
			'ta_state_hierarchy': self.ta_state_hierarchy_gpu.get(),
			'clause_weights': self.clause_weights_gpu.get(),
			'number_of_outputs': self.number_of_outputs,
			'min_y': self.min_y,
			'max_y': self.max_y,
			'params': {
				'number_of_clauses': self.number_of_clauses,
				'T': self.T,
				's': self.s,
				'q': self.q,
				'hierarchy_structure': self.hierarchy_structure,
				'boost_true_positive_feedback': self.boost_true_positive_feedback,
				'number_of_state_bits': self.number_of_state_bits,
				'append_negated': self.append_negated,
			},
			'negative_clauses': self.negative_clauses,
			'tm_type': self.tm_type,
			'flip_polarity': self.flip_polarity,
			'weighted_clauses': getattr(self, 'weighted_clauses', None),
		}

	def load(self, state_dict: dict):
		self.number_of_outputs = state_dict['number_of_outputs']
		self.min_y = state_dict['min_y']
		self.max_y = state_dict['max_y']

		self.allocate_gpu_memory()

		self.ta_state_hierarchy_gpu = cp.asarray(state_dict['ta_state_hierarchy'], dtype=cp.uint32)
		self.clause_weights_gpu = cp.asarray(state_dict['clause_weights'], dtype=cp.int32)

		self.first = False

	
class MultiOutputTsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, hierarchy_structure=((AND_GROUP, 1)), q=1.0, log_scale=False, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1), seed=None):
		self.negative_clauses = 1
		super().__init__(number_of_clauses, T, s, hierarchy_structure, q=q, log_scale=log_scale, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block, seed=seed)

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_outputs = Y.shape[1]
		self.patch_dim = (X.shape[1], 1, 1)

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)
		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def score(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		return self._score(X)

	def predict(self, X):
		return (self.score(X) >= 0).astype(np.uint32).transpose()

class MultiClassCoalescedTsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, q=1.0, log_scale=False, hierarchy_structure=((AND_GROUP, 1)), boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1), seed=None):
		self.negative_clauses = 1
		self.tm_type = COALESCED_TM
		self.flip_polarity = 1

		super().__init__(number_of_clauses, T, s, q=q, log_scale=log_scale, hierarchy_structure=hierarchy_structure, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block, seed=seed)

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_outputs = int(np.max(Y) + 1)
		self.patch_dim = (X.shape[1], 1, 1)

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype = np.int32)
		for i in range(self.number_of_outputs):
			encoded_Y[:,i] = np.where(Y == i, self.T, -self.T)

		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def score(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		return self._score(X)

	def predict(self, X):
		return np.argmax(self.score(X), axis=0)

class MultiClassTsetlinMachine:
	def __init__(self, number_of_clauses, T, s, q=1.0, log_scale=False, weighted_clauses=False, hierarchy_structure=((AND_GROUP, 1)), boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1), seed=None):
		self.number_of_clauses = number_of_clauses
		self.T = T
		self.s = s
		self.q = q
		self.log_scale = log_scale
		self.weighted_clauses = weighted_clauses
		self.hierarchy_structure = hierarchy_structure
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.number_of_state_bits = number_of_state_bits
		self.append_negated = append_negated
		self.grid = grid
		self.block = block
		self.seed = np.random.randint(1, 2**30) if seed is None else seed

		self.configured = False

	def fit(self, X, Y, epochs=100, incremental=False):
		self.number_of_outputs = int(np.max(Y) + 1)

		if not self.configured:
			self.tms = []
			for i in range(self.number_of_outputs):
				self.tms.append(TsetlinMachine(self.number_of_clauses, self.T, self.s, q=self.q, log_scale=self.log_scale, weighted_clauses=self.weighted_clauses, hierarchy_structure=self.hierarchy_structure, boost_true_positive_feedback=self.boost_true_positive_feedback, number_of_state_bits=self.number_of_state_bits, append_negated=self.append_negated, grid=self.grid, block=self.block, seed=self.seed+i))

			self.configured = True

		encoded_Y = np.empty(Y.shape[0], dtype = np.int32)

		for epoch in range(epochs):
			for i in range(self.number_of_outputs):
				target_X = X[Y==i]

				not_target_X = X[Y!=i]
				not_target_index = np.random.rand(not_target_X.shape[0]) <= 1.0/(self.number_of_outputs - 1)

				balanced_X = np.vstack((target_X, not_target_X[not_target_index,:]))
				balanced_Y = np.hstack((np.ones(target_X.shape[0]), np.zeros(not_target_X.shape[0])))
				index = np.arange(balanced_X.shape[0])
				np.random.shuffle(index)

				self.tms[i].fit(balanced_X[index], balanced_Y[index], epochs=1, incremental=incremental)
		return

	def score(self, X):
		class_sums = np.empty((self.number_of_outputs, X.shape[0]), dtype=np.int32)
		for i in range(self.number_of_outputs):
			class_sums[i,:] = self.tms[i].score(X)

		return class_sums

	def predict(self, X):
		return np.argmax(self.score(X), axis=0)

	def save(self) -> dict:
		return {
			'number_of_outputs': self.number_of_outputs,
			'tms': [t.save() for t in self.tms],
			'params': {
				'number_of_clauses': self.number_of_clauses,
				'T': self.T,
				's': self.s,
				'q': self.q,
				'weighted_clauses': self.weighted_clauses,
				'hierarchy_structure': self.hierarchy_structure,
				'boost_true_positive_feedback': self.boost_true_positive_feedback,
				'number_of_state_bits': self.number_of_state_bits,
				'append_negated': self.append_negated,
				'grid': self.grid,
				'block': self.block,
			},
		}

	def load(self, state_dict: dict):
		self.number_of_outputs = state_dict['number_of_outputs']

		self.tms = []
		for tm_state in state_dict['tms']:
			t = TsetlinMachine(self.number_of_clauses, self.T, self.s, weighted_clauses=self.weighted_clauses, hierarchy_structure=self.hierarchy_structure, q=self.q, boost_true_positive_feedback=self.boost_true_positive_feedback, number_of_state_bits=self.number_of_state_bits, append_negated=self.append_negated, grid=self.grid, block=self.block)
			t.load(tm_state)
			self.tms.append(t)

		self.configured = True

class TsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, q=1.0, log_scale=False, weighted_clauses=False, hierarchy_structure=((AND_GROUP, 1)), boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1), seed=None):
		self.negative_clauses = 1
		self.flip_polarity = 0

		if weighted_clauses:
			self.tm_type = WEIGHTED_TM
		else:
			self.tm_type = VANILLA_TM

		super().__init__(number_of_clauses, T, s, q=q, log_scale=log_scale, hierarchy_structure=hierarchy_structure, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block, seed=seed)

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_outputs = 1

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)

		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def score(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		return self._score(X)[0,:]

	def predict(self, X):
		return (self.score(X) >= 0).astype(np.int32)

class RegressionTsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, log_scale=False, hierarchy_structure=((AND_GROUP, 1)), boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1), seed=None):
		self.negative_clauses = 0
		self.flip_polarity = 0

		super().__init__(number_of_clauses, T, s, log_scale=log_scale, hierarchy_structure=hierarchy_structure, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block, seed=seed)

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		
		self.number_of_outputs = 1
		self.patch_dim = (X.shape[1], 1, 1)

		self.max_y = np.max(Y)
		self.min_y = np.min(Y)
	
		encoded_Y = ((Y - self.min_y)/(self.max_y - self.min_y)*self.T).astype(np.int32)
			
		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def predict(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		
		return 1.0*(self._score(X)[0,:])*(self.max_y - self.min_y)/(self.T) + self.min_y
