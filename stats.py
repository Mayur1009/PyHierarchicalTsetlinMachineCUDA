import numpy as np
import sys

m = np.loadtxt(sys.argv[1])

runs = int(m[:,0].max())

for run in range(int(runs)):
	i = m[m[:,0]==run][:,2].argmax()
	print(m[m[:,0]==run][i])