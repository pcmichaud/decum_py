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
        #data = data.sample(n = 270)
        maxeval=10000
        pars = np.load('output/estimates_ez.npy')
        pars[4] = 0.0
        isfree = np.ones(pars.shape[0])
        #isfree[1] = 0
        #isfree[2] = 0
        #isfree[3] = 0
        isfree[4] = 0
        #isfree[5] = 0
        #isfree[6] = 0
        #isfree[7] = 0
        #isfree[8] = 0
        #isfree[9] = 0
        theta = set_theta(pars,isfree)
        n_free_theta = theta.shape[0]
        print('number of parameters = ',pars.shape[0],', number of free parameters = ',n_free_theta)
        dx = np.zeros(n_free_theta)
        for i in range(n_free_theta):
                dx[i] = np.abs(theta[i])*0.25
        partial_distance = partial(concentrated_distance_within,data=data,isfree=isfree,ipars=pars,npartitions=256, iwithin=False,scn_name='ez_levels')
        opt = nl.opt('LN_NEWUOA',n_free_theta)
        opt.set_min_objective(partial_distance)
        opt.set_initial_step(dx)
        opt.set_maxeval(maxeval)
        opt.set_xtol_abs(1e-5)
        theta = opt.optimize(theta)
        distance = opt.last_optimum_value()
        opt_pars = extract_pars(theta, isfree, pars)
        print('estimates = ', opt_pars)
        print('final distance = ', distance)
        np.save('output/estimates_ez_levels',opt_pars)

