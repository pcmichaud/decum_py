from optim import *
import warnings
from numba import set_num_threads
import pandas as pd
import nlopt  as nl
from functools import partial

if __name__ == '__main__':
	warnings.simplefilter(action='ignore')
	data = init_data()
	#data = data.sample(n=100)
	n_part = 250
	# estimates from reference scenario
	sigmas = np.load('output/sigmas_ref.npy')
	pars = np.load('output/estimates_ref.npy')
	isfree = np.ones(pars.shape[0])
	theta = set_theta(pars,isfree)
	# sizes
	n_free_theta = theta.shape[0]
	n_sigmas = sigmas.shape[0]*sigmas.shape[1]
	J = n_free_theta + n_sigmas
	nresp = len(data)
	K = 12
	eps = 1e-4
	es = residuals_within(theta, sigmas, data, isfree, pars, npartitions=n_part)
	es.to_csv('output/within_residuals_ref.csv')
	es = es.stack()
	print(es.head(50))
	theta = set_theta(pars,isfree)
	gs = g_within(theta, sigmas, data, isfree, pars, npartitions=n_part).stack()
	grad = pd.DataFrame(index=es.index,columns=[x for x in range(J)])
	for j in range(n_free_theta):
		print(j)
		theta_up = theta[:]
		theta_up[j] += eps
		gs_up = g_within(theta_up, sigmas, data, isfree, pars, npartitions=n_part).stack()
		grad[j] = (gs_up - gs)/eps
	print(grad.head(50))
	j_start = n_free_theta
	jj = 0
	for j in range(3):
		for k in range(2):
			sigmas_up = sigmas[:,:]
			sigmas_up[j,k] += eps
			gs_up = g_within(theta, sigmas_up, data, isfree, pars, npartitions=n_part).stack()
			grad[j_start+jj] = (gs_up - gs)/eps
			jj +=1
	print(grad.head(50))
	grad.to_csv('output/gradients_ref.csv')





