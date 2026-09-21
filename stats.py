import numpy as np
import sys

m = np.loadtxt(sys.argv[1])

print(m)

runs = int(m[:,0])

for run in range(int(runs)):
	print(m[m[:,0]==run][:,3].max())