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
        #data = data.sample(n=250, random_state=1234)
        maxeval=10000
        pars = np.array([5.89018885,
                         2.29888801,
                         0.96276999,
                         0.07099994,
                        0.,
                        1.,
                        0.13007304,
                        0.31885028])
        isfree = np.ones(pars.shape[0])
        #isfree[0] = 0
        #isfree[1] = 0
        #isfree[2] = 0
        #isfree[3] = 0
        isfree[4] = 0
        isfree[5] = 0
        #isfree[6] = 0
        #isfree[7] = 0

        theta = set_theta(pars,isfree)
        n_free_theta = theta.shape[0]
        print('number of parameters = ',pars.shape[0],', number of free parameters = ',n_free_theta)
        dx = np.zeros(n_free_theta)
        dx[0] = 0.25
        dx[1] = 0.25
        dx[2] = 0.01
        dx[3] = 0.25
        dx[4] = 0.25
        dx[5] = 0.25
        #dx[4] = 0.2
        #dx[5] = 0.2
        print('theta = ',theta)
        print('dx = ',dx)
        partial_distance = partial(concentrated_distance_within,data=data,isfree=isfree,ipars=pars,npartitions=256, iwithin = False,  iann = True, irmr= True, iltc = True, scn_name='levels')
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
        np.save('output/estimates_levels',opt_pars)

