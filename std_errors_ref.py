from optim import *
import warnings
from numba import set_num_threads
import pandas as pd
import nlopt  as nl
from functools import partial




if __name__ == '__main__':
	warnings.simplefilter(action='ignore')
	data = init_data()
	data = data.sample(n=250)
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
	eps = 1e-4

	es = residuals_within(theta, sigmas, data, npartitions=250).stack()

	print(es.head(50))

	grad = pd.DataFrame(index=es.index,columns=[x for x in range(J)])
	
	for j in range(n_free_theta):
		print(j)
		pars_up = pars[:]
		pars_up[j] += eps
		theta_up = set_theta(pars_up)
		es_up = residuals_within(theta_up, sigmas, data, npartitions=250).stack()
		grad[j] = (es_up - es)/eps

	print(grad.head(50))	
	j_start = n_free_theta 
	theta = set_theta(pars)
	for j in range(3):
		print(j)
		sigmas_up = sigmas[:]
		sigmas_up[j] += eps 
		es_up = residuals_within(theta, sigmas_up, data, npartitions=250).stack()
		grad[j_start+j] = (es_up - es)/eps
	print(grad.head(50))

	A = np.zeros((J,J))
	B = np.zeros((J,J))
	for i in data.index:
		e_i = es.loc[i,:].to_numpy()
		e_i[np.isnan(e_i)] = 0
		g_i = grad.loc[(i,),:].to_numpy()
		g_i[np.isnan(g_i)] = 0
		A += g_i.T @ g_i
		B += 






