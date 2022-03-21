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
			0.5,
			0.108,
			0.515,
			0.001,
			0.0007,
			2617.0,
			0.669,
			0.009,
			0.096,
			0.019,
			0.004,
			-0.835,
			0.601])
	partial_distance = partial(concentrated_distance_within,data=data,npartitions=250)
	eps = np.linspace(0.5,5.0,5)
	ssds = []
	for ep in eps: 
		pars_g = pars[:]
		pars_g[0] = ep
		theta = set_theta(pars)
		n_free_theta = theta.shape[0]
		grad = 0.0
		ssd = partial_distance(theta,grad)
		ssds.append(ssd)
		print(ep,ssd)

		


