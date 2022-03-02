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
	pars = np.array([2.0,
			0.2,
			0.03,
			0.02,
			0.02,
			3.0,
			0.2,
			11.0,
			0.8,
			0.002,
			0.05,
			0.03,
			0.03,
			-1.73,
			-3.0])
	theta = set_theta(pars)
	n_free_theta = theta.shape[0]
	dx = np.zeros(n_free_theta)
	for i in range(n_free_theta):
		dx[i] = np.abs(theta[i])*0.2
	partial_distance = partial(concentrated_distance_levels,data=data,npartitions=250)
	opt = nl.opt('LN_NEWUOA',n_free_theta)
	opt.set_min_objective(partial_distance)
	opt.set_initial_step(dx)
	opt.set_maxeval(maxeval)
	opt.set_xtol_abs(1e-4)
	theta = opt.optimize(theta)
	distance = opt.last_optimum_value()
	opt_pars = extract_pars(theta)
	print('estimates = ', opt_pars)
	print('final distance = ', distance)
