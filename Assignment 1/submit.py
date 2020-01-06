import numpy as np
import random as rnd
import time as tm

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
# Umesh Meena
# Tushar
# Varun
# Ganesh
# Vishal


def sq_hinge_func(X,Y,W):
    return (np.maximum(1-np.multiply(X.dot(W),Y),0))**2

def loss_func(X,Y,W,C=1):
#     print(W.shape,)
    loss = 0.5*W.dot(W)+ C*sum(sq_hinge_func(X,Y,W))
    return loss

def getStepLength( eta, t ):
    return eta/(t+1)

def sq_hinge_gradient(X,Y,W):
	n = Y.shape[0]	
	pred = X.dot(W)
	gradients = np.zeros((n,))
	margin = np.multiply(pred,Y)
	gradients[margin < 1] = -2*(1-margin[margin<1])

	return gradients

def mini_batch_SGD(X,Y,n,_iter,W,C=1,eta=0.01):
    
    sq_hinge_gr = sq_hinge_gradient(X,Y,W)  
    batch_size = X.shape[0]
    gradient = W + C * (n/batch_size)*(X.T).dot(np.multiply(Y,sq_hinge_gr))
    
    W = W -getStepLength(eta,_iter)*gradient
    return W

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
	eta=0.0005
	batch_size=512

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
				return (w, b, totTime)
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

		w = np.insert(w, 0,b,0)
		if (batch_size < n):
			tr_index = np.random.choice(range(0,n),batch_size)
			X_train,Y_train = X_tr[tr_index,:],y[tr_index]
		else:
			X_train,Y_train = X_tr,y
		w = mini_batch_SGD(X_train,Y_train,n,t,w,C,eta)

		b= w[0]
		w = w[1:]

	return (w, b, totTime) # This return statement willd+j never be reached