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
	data = data.sample(n=250)
	maxeval=10000
	pars = np.array([0.614,
			0.016,
			1.584,
			0.108,
			0.515,
			0.001,
			0.0007,
			2617.0,
			0.669,
			0.009,
			0.096,
			0.004,
			-0.835,
			0.601])
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
        #np.save('output/estimates_test.npy',opt_pars)

