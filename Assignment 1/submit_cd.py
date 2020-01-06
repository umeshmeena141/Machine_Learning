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
	
def sq_hinge_func(X,Y,W):
    return (np.maximum(1-np.multiply(X.dot(W),Y),0))**2

def loss_func(X,Y,W,C=1):
#     print(W.shape,)
    loss = 0.5*W.dot(W)+ C*sum(sq_hinge_func(X,Y,W))
    return loss

def getStepLength( eta, t ):
    return eta/np.sqrt(t+1)

def sq_hinge_gradient(X,Y,W,pred):
    n = Y.shape[0]
    gradients = np.zeros((n,))
    margin = np.multiply(pred,Y)
    gradients[margin < 1] = -2*(1-margin[margin<1])
    
    return gradients

def SCD(X,Y,coord,pred,_iter,W,C=1,eta=0.01):
    
    sq_hinge_gr = sq_hinge_gradient(X,Y,W,pred)  
#     batch_size = X.shape[0]
#     print(coord)
    gradient = W[coord] + C*(X.T)[coord].dot(np.multiply(Y,sq_hinge_gr))

    pred -= W[coord]*X[:,coord]
    W[coord] = W[coord]- getStepLength(eta,_iter)*gradient
    pred += W[coord]*X[:,coord]
    return W,pred
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
	w = np.zeros( (d,) )
	X_tr = np.hstack((np.ones((n,1)),X))
	eta=0.0001
	_iter=3000
	prev=-1
	timeSeries = np.array( [] )
	_totTime = 0
	primalObjValSeries = np.array([])
	# You may reinitialize w, b to your liking here
	# You may also define new variables here e.g. eta, B etc

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
		# if(t>_iter): return w,b,totTime
		_tic = tm.perf_counter()

		w = np.insert(w, 0, b, axis=0)
		# w_prev = np.copy(w)
		pred = X_tr.dot(w)
		j = getCyclicCoord(prev,d)
		prev = j
		w,pred = SCD(X_tr,y,j,pred,t,w,C,eta)
		_toc = tm.perf_counter()

		_totTime = _totTime + (_toc - _tic)
		primalObjValSeries = np.append(primalObjValSeries ,loss_func(X_tr,y, w ,C ))
		timeSeries = np.append(timeSeries,_totTime)

		b= w[0]
		w = w[1:]
			
	return (w, b, totTime,timeSeries,primalObjValSeries) # This return statement willd+j never be reached