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
    sigmas = np.load('output/sigmas_ez.npy')
    sigmas = sigmas[:,0]
    pars = np.load('output/estimates_ez.npy')
    isfree = np.ones(pars.shape[0])
    isfree[4] = 0
    isfree[5] = 0
    theta = set_theta(pars,isfree)
    # sizes
    n_free_theta = theta.shape[0]
    n_sigmas = sigmas.shape[0]
    J = n_free_theta + n_sigmas
    nresp = len(data)
    eps = 1e-5
    es = residuals_within(theta, sigmas, data, isfree, pars, npartitions=n_part)
    es.to_csv('output/within_residuals_ez.csv')
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
        theta_low = theta[:]
        theta_low[j] -= eps
        gs_low = g_within(theta_low, sigmas, data, isfree, pars, npartitions=n_part).stack()
        grad[j] = (gs_up - gs_low)/(2.0*eps)
    j_start = n_free_theta
    for j in range(3):
        s = np.zeros(3)
        s[j] = 1
        gs = g_within(theta, s, data, isfree, pars, npartitions=n_part).stack()
        grad[j_start+j] = gs
    print(grad.head(50))
    print(J)
    grad.to_csv('output/gradients_ez.csv')





