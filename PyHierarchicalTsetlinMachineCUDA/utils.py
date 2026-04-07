from collections import deque
import numpy as np
import networkx as nx

from .tm import OR_ALTERNATIVES, CommonTsetlinMachine


def make_hierarchy_graph(G: nx.Graph, hier: list[tuple[int, int]]):
	"""Create Graph from the hierarchy."""
	# BFS
	q = deque()
	q.append((len(hier), 0))  # (level, idx)
	while q:
		level, idx = q.popleft()
		G.add_node((level, idx), label=hier[level - 1][0], op=hier[level - 1][0])
		if level == 1:
			continue
		branching = hier[level - 1][1]
		for cid in range(idx * branching, (idx + 1) * branching):
			q.append((level - 1, cid))
			G.add_edge((level, idx), (level - 1, cid))

def clause_to_nx(
	tm: CommonTsetlinMachine,
	clause_idx: int,
	feature_names: list[str] | None = None,
	negation_prefix: str = '¬',
	clause_literals: np.ndarray | None = None,
	ta_to_fid_mapping = None,
):
	"""
	Create a networkx Graph for a single clause.
	Args:
		`tm`: The model.
		`clause_idx`: Index of the clause to visualize.
		`feature_names`: Optional list of feature names for labeling. If None, defaults to 'x0', 'x1', ...
		`negation_prefix`: Prefix for negated features (default: '¬').
		`clause_literals`: Optional pre-extracted literals for the clause, must have shape (n_components, literals_per_component). If None, literals will be extracted from the model.
		`ta_to_fid_mapping`: Optional mapping from (component_id, literal_id) to feature_id. If None, it will be obtained from the model.
	"""

	assert tm.hierarchy_structure is not None, "Hierarchy structure not defined in the model. Are you sure tm belongs to CommonTsetlinMachine or its subclasses?"

	if feature_names is None:
		feature_names = [f'x{i}' for i in range(tm.number_of_features_hierarchy)]

	literals = clause_literals if clause_literals is not None else tm.get_literals()[clause_idx]
	ta_to_fid = ta_to_fid_mapping if ta_to_fid_mapping is not None else tm.map_ta_id_to_feature_id()
	n_comp = tm.hierarchy_size[1]
	lits_per_comp = tm.number_of_literals_per_leaf

	G = nx.Graph()
	make_hierarchy_graph(G, tm.hierarchy_structure)

	feat_per_comp = lits_per_comp // 2 if tm.append_negated else lits_per_comp

	# Add included literals to leaf components
	for comp_id in range(n_comp):
		for fid in range(feat_per_comp):
			pos_lit = literals[comp_id, fid]
			if pos_lit:
				G.add_node(
					(0, comp_id * lits_per_comp + fid),
					label=feature_names[ta_to_fid[comp_id, fid]],
				)
				G.add_edge((1, comp_id), (0, comp_id * lits_per_comp + fid))

			if tm.append_negated:
				neg_lit = literals[comp_id, feat_per_comp + fid]
				if neg_lit:
					lab = f'{negation_prefix}{feature_names[ta_to_fid[comp_id, fid]]}'
					G.add_node(
						(0, comp_id * lits_per_comp + feat_per_comp + fid), label=lab
					)
					G.add_edge(
						(1, comp_id), (0, comp_id * lits_per_comp + feat_per_comp + fid)
					)

	return G


def clause_bank_to_nx(
	tm: CommonTsetlinMachine,
	feature_names: list[str] | None = None,
	negation_prefix: str = '¬',
):
	"""
	Create a networkx Graph for the entire clause bank.
	Args:
		`tm`: The model.
		`feature_names`: Optional list of feature names for labeling. If None, defaults to 'x0', 'x1', ...
		`negation_prefix`: Prefix for negated features (default: '¬').
	"""
	literals = tm.get_literals()
	ta_to_fid = tm.map_ta_id_to_feature_id()
	G = nx.Graph()
	cb_root = "CB_ROOT"
	G.add_node(cb_root, label=OR_ALTERNATIVES, op=OR_ALTERNATIVES)
	for ci in range(tm.number_of_clauses):
		clause_G = clause_to_nx(tm, ci, feature_names, negation_prefix, literals[ci], ta_to_fid)
		mapping = { node: (ci, node) for node in clause_G.nodes }
		G = nx.compose(G, nx.relabel_nodes(clause_G, mapping))
		G.add_edge(cb_root, (ci, (tm.depth, 0)))

	return G
