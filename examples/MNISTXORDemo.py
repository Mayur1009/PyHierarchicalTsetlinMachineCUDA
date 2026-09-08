from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
from keras.datasets import mnist

clauses = 4
T = 100*100
s = 10.0

number_of_training_examples = 10000
number_of_testing_examples = 1000

(X_mnist_train, Y_mnist_train), (X_mnist_test, Y_mnist_test) = mnist.load_data()

X_mnist_train = np.where(X_mnist_train.reshape((X_mnist_train.shape[0], 28*28)) > 75, 1, 0)
X_mnist_test = np.where(X_mnist_test.reshape((X_mnist_test.shape[0], 28*28)) > 75, 1, 0)

X_train = np.empty((number_of_training_examples, 28*28*2))
Y_train = np.empty(number_of_training_examples)
for i in range(number_of_training_examples):
	x = np.random.randint(2, size=(2))

	X_train[i,:28*28] = X_mnist_train[Y_mnist_train == x[0]][0]
	X_train[i,28*28:] = X_mnist_train[Y_mnist_train == x[1]][0]	

	Y_train[i] = np.logical_xor(x[0], x[1])


X_test = np.empty((number_of_testing_examples, 28*28*2))
Y_test = np.empty(number_of_testing_examples)
for i in range(number_of_testing_examples):
	x = np.random.randint(2, size=(2))

	X_test[i,:28*28] = X_mnist_train[Y_mnist_train == x[0]][0]
	X_test[i,28*28:] = X_mnist_train[Y_mnist_train == x[1]][0]	

	Y_test[i] = np.logical_xor(x[0], x[1])

tm = MultiClassTsetlinMachine(clauses, T, s, hierarchy_structure=((tm.AND_GROUP, 28*28), (tm.OR_ALTERNATIVES, 100), (tm.AND_GROUP, 2)))

print("\nAccuracy over 500 epochs:\n")
for i in range(500):
	start_training = time()
	tm.fit(X_train, Y_train)
	stop_training = time()


	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
