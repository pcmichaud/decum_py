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
	#data = data.sample(n=250)
	maxeval=10000
	pars = np.array([0.614,
			0.016,
			0.65,
			0.108,
			0.65,
			0.05,
			0.05,
			500.0,
			0.669,
			0.1,
			0.1,
			0.05,
			0.1,
			0.1])
	isfree = np.ones(pars.shape[0])
	theta = set_theta(pars,isfree)
	n_free_theta = theta.shape[0]
	print('number of parameters = ',pars.shape[0],', number of free parameters = ',n_free_theta)
	dx = np.zeros(n_free_theta)
	for i in range(n_free_theta):
		dx[i] = np.abs(theta[i])*0.25
	partial_distance = partial(concentrated_distance_within,data=data,isfree=isfree,ipars=pars,npartitions=250, scn_name='ref')
	opt = nl.opt('LN_NEWUOA',n_free_theta)
	opt.set_min_objective(partial_distance)
	opt.set_initial_step(dx)
	opt.set_maxeval(maxeval)
	opt.set_xtol_abs(1e-4)
	theta = opt.optimize(theta)
	distance = opt.last_optimum_value()
	opt_pars = extract_pars(theta, isfree, pars)
	print('estimates = ', opt_pars)
	print('final distance = ', distance)
	np.save('output/estimates_ref',opt_pars)

