#!/usr/bin/python

# equivalent implementation of the simple 2D example, in Python.

from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import ipdb as pdb

TRAJECTORY = np.zeros((0,2))

def simple2d_q(x):
	# velocity field
	F = simple2D_F(x)
	q = 0.5*(np.sum(F**2))
	return q

def simple2D_F(x):
	return np.array([(1-x[0]**2)*x[1], x[0]/2-x[1]])

def simple2d_jac(x):
	return np.array([
		[-2*x[0]*x[1],1-x[0]**2],
		[1/2,-1]
	])

def simple2d_grad(x):
	F = simple2D_F(x)
	J = simple2d_jac(x)
	return np.dot(J,F)

def simple2d_hess(x):
	J = simple2d_jac(x)
	return np.dot(J.transpose(),J)

def simple2d_callback(xk):
	global TRAJECTORY
	TRAJECTORY = np.vstack((TRAJECTORY, xk))


if __name__ == "__main__":
	xmin = -1.5;
	xmax = 1.5;

	nTrajectories = 50
	plt.figure()

	xx = np.arange(xmin,xmax,.1)
	(xs,ys) = np.meshgrid(xx,xx)

	U = np.zeros(xs.shape)
	V = np.zeros(xs.shape)
	for i in range(xs.shape[0]):
		for j in range(xs.shape[1]):
			(U[i][j], V[i][j]) = simple2D_F([xs[i][j],ys[i][j]])

	plt.quiver(xs,ys,U,V)

	for i in range(nTrajectories):
		TRAJECTORY = np.zeros((0,2)) # reset it
		x0 = np.random.uniform(low=xmin,high=xmax,size=2)
		#x0 = np.array([-1.5,1.5]) + 0.3*np.random.rand()
		res = minimize(simple2d_q, x0, method='Newton-CG', # trust-region doesn't seem to work
			jac=simple2d_grad, hess=simple2d_hess,
			options={'xtol': 1e-20, 'disp': False},
			callback=simple2d_callback)
		X = TRAJECTORY[:,0]
		Y = TRAJECTORY[:,1]
		plt.plot(X,Y)
	plt.xlim((xmin,xmax))
	plt.ylim((xmin,xmax))
	plt.show()
