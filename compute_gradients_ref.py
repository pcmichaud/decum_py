from optim import *
import warnings
from numba import set_num_threads
import pandas as pd
import nlopt  as nl
from functools import partial




if __name__ == '__main__':
	warnings.simplefilter(action='ignore')
	data = init_data()
	#data = data.sample(n=50)
	n_part = 250
	# estimates from reference scenario
	sigmas =  np.array([0.03203350286930366, 0.4488275893091808, 0.06445042119004554])
	pars =  np.array([ 6.14015775e-01,  1.60347627e-02,  5.83926854e-02,  1.10812720e-01,
  		5.15115871e-01,  1.23220592e-03,  7.24679259e-04,  2.61065057e+03,
  		6.69294074e-01,  9.40945610e-03,  9.59258511e-02,  1.91233141e-02,
  		4.15400655e-03, -8.35699053e-01,  6.00580451e-01])
	theta = set_theta(pars)
	# sizes
	n_free_theta = theta.shape[0]
	n_sigmas = 3
	J = n_free_theta + n_sigmas
	nresp = len(data)
	K = 12
	eps = 1e-5
	es = residuals_within(theta, sigmas, data, npartitions=n_part)
	es.to_csv('output/within_residuals_ref.csv')
	es = es.stack()

	print(es.head(50))

	theta = set_theta(pars)
	gs = g_within(theta, sigmas, data, npartitions=n_part).stack()
	grad = pd.DataFrame(index=es.index,columns=[x for x in range(J)])
	for j in range(n_free_theta):
		print(j)
		theta_up = theta[:]
		theta_up[j] += eps
		gs_up = g_within(theta_up, sigmas, data, npartitions=n_part).stack()
		grad[j] = (gs_up - gs)/eps
	print(grad.head(50))
	j_start = n_free_theta
	for j in range(n_sigmas):
		print(j)
		sigmas_up = sigmas[:]
		sigmas_up[j] += eps
		gs_up = g_within(theta, sigmas_up, data, npartitions=n_part).stack()
		grad[j_start+j] = (gs_up - gs)/eps
	print(grad.head(50))
	grad.to_csv('output/gradients_ref.csv')





