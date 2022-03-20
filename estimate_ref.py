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
	#data = data.sample(n=500)
	maxeval=10000
	pars = np.array([0.5,
			0.1,
			0.25,
			0.25,
			0.735,
			0.042,
			0.061,
			517.0,
			0.546,
			0.377,
			0.21,
			0.068,
			0.02,
			-0.094,
			0.082])
	theta = set_theta(pars)
	n_free_theta = theta.shape[0]
	dx = np.zeros(n_free_theta)
	for i in range(n_free_theta):
		dx[i] = np.abs(theta[i])*0.25
	partial_distance = partial(concentrated_distance_within,data=data,npartitions=250)
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
        np.save('output/estimates.npy',opt_pars)

