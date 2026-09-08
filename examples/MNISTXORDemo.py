from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
from keras.datasets import mnist

or_alternatives = 4

clauses = 4
T = 0.8*or_alternatives*or_alternatives*4
s = 20.0

number_of_training_examples = 10000
number_of_testing_examples = 1000

(X_mnist_train, Y_mnist_train), (X_mnist_test, Y_mnist_test) = mnist.load_data()

X_mnist_train = np.where(X_mnist_train.reshape((X_mnist_train.shape[0], 28*28)) > 75, 1, 0)
X_mnist_test = np.where(X_mnist_test.reshape((X_mnist_test.shape[0], 28*28)) > 75, 1, 0)

x_mnist_train_count = [X_mnist_train[Y_mnist_train == 0].shape[0], X_mnist_train[Y_mnist_train == 1].shape[0]]

X_train = np.empty((number_of_training_examples, 28*28*2))
Y_train = np.empty(number_of_training_examples)
for i in range(number_of_training_examples):
	x = np.random.randint(2, size=(2))

	X_train[i,:28*28] = X_mnist_train[Y_mnist_train == x[0]][np.random.randint(x_mnist_train_count[x[0]])]
	X_train[i,28*28:] = X_mnist_train[Y_mnist_train == x[1]][np.random.randint(x_mnist_train_count[x[1]])]	

	Y_train[i] = np.logical_xor(x[0], x[1])

np.savetxt("examples/MNISTXORTrainingData.txt", np.append(X_train, Y_train.reshape((number_of_training_examples, 1)), axis=1), fmt='%d')

X_test = np.empty((number_of_testing_examples, 28*28*2))
Y_test = np.empty(number_of_testing_examples)
for i in range(number_of_testing_examples):
	x = np.random.randint(2, size=(2))

	X_test[i,:28*28] = X_mnist_train[Y_mnist_train == x[0]][np.random.randint(x_mnist_train_count[x[0]])]
	X_test[i,28*28:] = X_mnist_train[Y_mnist_train == x[1]][np.random.randint(x_mnist_train_count[x[1]])]	

	Y_test[i] = np.logical_xor(x[0], x[1])

np.savetxt("examples/MNISTXORTestingData.txt", np.append(X_test, Y_test.reshape((number_of_testing_examples, 1)), axis=1), fmt='%d')

train_data = np.loadtxt("./examples/MNISTXORTrainingData.txt").astype(np.uint32)
X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

test_data = np.loadtxt("./examples/MNISTXORTestingData.txt").astype(np.uint32)
X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

tm = MultiClassTsetlinMachine(clauses, T, s, hierarchy_structure=((tm.AND_GROUP, 28*28), (tm.OR_ALTERNATIVES, or_alternatives), (tm.AND_GROUP, 2)))

print("\nAccuracy over 500 epochs:\n")
for i in range(500):
	start_training = time()
	tm.fit(X_train, Y_train)
	stop_training = time()


	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
