import numpy as np
import matplotlib.pyplot as plt
from submit import solver 
# from submit_sdcm import solver as solver_sdcm
# from submit_cd import solver as solver_cd


def getObj( X, y, w, b ):
	hingeLoss = np.maximum( 1 - np.multiply( (X.dot( w ) + b), y ), 0 )
	return 0.5 * w.dot( w ) + C * hingeLoss.dot( hingeLoss )

Z = np.loadtxt( "data" )

y = Z[:,0]
X = Z[:,1:]
C = 1

avgTime = 0
avgPerf = 0

# To avoid unlucky outcomes try running the code several times
numTrials = 5
# 30 second timeout for each run
timeout = 3
# Try checking for timeout every 100 iterations
spacing = 100

for t in range( numTrials ):
	(w, b, totTime) = solver( X, y, C, timeout, spacing )
	avgTime = avgTime + totTime
	avgPerf = avgPerf + getObj( X, y, w, b )

# print(avgPerf,avgTime)
print( avgPerf/numTrials, avgTime/numTrials )