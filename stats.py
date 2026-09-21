import numpy as np
import sys

m = np.loadtxt(sys.argv[1])

print(m)

runs = m[:,0]
print(runs.max())