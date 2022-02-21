from optim import *
import warnings
from numba import set_num_threads
import pandas as pd 
import nlopt  as nl
from functools import partial

if __name__ == '__main__':
	warnings.simplefilter(action='ignore')
	data = init_data()
	data = data.iloc[:500,:]
	maxeval=10000
	pars = np.array([4.0,
			0.5,
			0.2,
			0.4,
			0.7,
			0.2,
			0.5,
			136.0,
			0.72,
			0.35,
			2.5,
			1.5,
			0.5,
			0.25,
			0.25])
	theta = set_theta(pars)
	n_free_theta = theta.shape[0]  
	dx = np.zeros(n_free_theta)
	for i in range(n_free_theta):
		dx[i] = np.abs(theta[i])*0.1
	partial_distance = partial(concentrated_distance,data=data,npartitions=250)
	opt = nl.opt('LN_NEWUOA',n_free_theta)
	opt.set_min_objective(partial_distance)
	opt.set_initial_step(dx)
	opt.set_maxeval(maxeval)
	opt.set_xtol_abs(1e-5)
	theta = opt.optimize(theta)
	distance = opt.last_optimum_value()
	opt_pars = extract_pars(theta)
	print('estimates = ', opt_pars)
	print('final distance = ', distance)