from optim import *
import warnings
from numba import set_num_threads
import pandas as pd
import nlopt  as nl
from functools import partial
import numpy as np

if __name__ == '__main__':
	warnings.simplefilter(action='ignore')
	data = init_data()
	data = data.sample(n=250,random_state=1234)
	maxeval=10000
	pars = np.array([0.614,
			0.016,
			0.5,
			0.1,
			0.515,
			0.05,
			0.05,
			500.0,
			0.6,
			0.2,
			0.1,
			0.025,
			-0.8,
			0.5])
	partial_distance = partial(concentrated_distance_within,data=data,npartitions=250)
	theta = set_theta(pars)
	grad = 0.0
	ssd = partial_distance(theta,grad)
	
	#eps = np.linspace(0.5,0.95,4)
	#gammas = np.linspace(0.5,0.95,4)
	#rhos = np.linspace(0.45,0.90,3)
	#bxs = np.linspace(0,0.5,5)
	#for ep in eps: 
""" 		for gamma in gammas:
			for rho in rhos:
				for bx in bxs:
					pars_g = pars[:]
					pars_g[0] = ep
					pars_g[2] = gamma 
					pars_g[4] = rho
					pars_g[5] = bx
					theta = set_theta(pars_g)
					grad = 0.0
					ssd = partial_distance(theta,grad)
					print('*** - ',ep,gamma,rho,bx, ssd) """

		


