import numpy as np
import random as rnd
import time as tm

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def getCyclicCoord( currentCoord,n ):
    if currentCoord >= n-1 or currentCoord < 0:
        return 0
    else:
        return currentCoord + 1
    
def getRandCoord( currentCoord,n ):
    return rnd.randint( 0, n-1 )

def getCSVMObjVal( X,y,w,C):
    hingeLoss = np.maximum( 1 - np.multiply( (X.dot( w )), y ), 0 )
    return 0.5 * w.dot( w ) + C * hingeLoss.dot( hingeLoss )

def getCSVMObjValDual( alpha, w,C ):
    # Recall that b is supposed to be treated as the last coordinate of w
#     return np.sum(alpha)- np.square(np.linalg.norm(alpha))/(4*C) - 0.5 * np.square( np.linalg.norm( w ) )
    return np.sum(alpha) - alpha.dot(alpha)/(4*C) - 0.5 * np.square( np.linalg.norm( w ) )
################################
# Non Editable Region Starting #
################################
def solver( X, y, C, timeout, spacing ):
	(n, d) = X.shape
	t = 0
	totTime = 0
	
	# w is the normal vector and b is the bias
	# These are the variables that will get returned once timeout happens
	w = np.zeros( (d,) )
	b = 0
	tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
	X_tr = np.hstack((np.ones((n,1)),X))

	alpha = np.ones((n,))
	alphay = np.multiply( alpha, y )
	w = X_tr.T.dot( alphay )
	normSq = np.square( np.linalg.norm( X_tr, axis = 1 ) ) + 1
	c_term = 1/(2*C)
	i = -1
	b= w[0]
	w = w[1:]
	timeSeries = np.array( [] )
	_totTime = 0
	primalObjValSeries = np.array([])
################################
# Non Editable Region Starting #
################################
	while True:
		t = t + 1
		if t % spacing == 0:
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:
				return (w, b, totTime,timeSeries,primalObjValSeries)
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

		# Write all code to perform your method updates here within the infinite while loop
		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses - severe penalties await
		
		# Please note that once timeout is reached, the code will simply return w, b
		# Thus, if you wish to return the average model (as we did for GD), you need to
		# make sure that w, b store the averages at all times
		# One way to do so is to define two new "running" variables w_run and b_run
		# Make all GD updates to w_run and b_run e.g. w_run = w_run - step * delw
		# Then use a running average formula to update w and b
		# w = (w * (t-1) + w_run)/t
		# b = (b * (t-1) + b_run)/t
		# This way, w and b will always store the average and can be returned at any time
		# w, b play the role of the "cumulative" variable in the lecture notebook
		# w_run, b_run play the role of the "theta" variable in the lecture notebook
		# if(t>n): return (w, b, totTime,timeSeries,primalObjValSeries)
		# print(t)s
		_tic = tm.perf_counter()

		w = np.append(b,w)
		i = getCyclicCoord( i,n )
		x = X_tr[i,:]
		newAlphai =  alpha[i]*normSq[i]/(normSq[i]+c_term) + (1 - y[i] * (x.dot(w))) / (normSq[i]+c_term)
		# print(newAlphai)
			# if newAlphai < C:
			# 	newAlphai = C
		if newAlphai < 0:
			newAlphai = 0
        
		w = w + (newAlphai - alpha[i]) * y[i] * x

		alpha[i] = newAlphai
		_toc = tm.perf_counter()

		_totTime = _totTime + (_toc - _tic)
		primalObjValSeries = np.append(primalObjValSeries ,getCSVMObjVal(X_tr,y, w ,C ))
		timeSeries = np.append(timeSeries,_totTime)

		# print(t)
		b= w[0]
		w = w[1:]
			
	return (w, b, totTime,timeSeries,primalObjValSeries) # This return statement willd+j never be reached