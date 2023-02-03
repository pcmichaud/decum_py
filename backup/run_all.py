from frame import *
import warnings
from numba import set_num_threads
import pandas as pd

if __name__ == '__main__':
    warnings.simplefilter(action='ignore')
    data = init_data()
    data['xi'] = 0.0
    data['xi_sp'] = 0.0
    data['mu'] = 1.0
    data['zeta'] = 1.0
    data['g'] = 0.0
    data['sig'] = 0.0
    theta = np.load('output/estimates_reference.npy')
    theta[3:4] = 0.0
    theta[6:7] = 1.0
    values = solve_df(data, npartitions=250,theta=theta)
    print(values[['value_'+str(x) for x in range(13)]].describe().transpose())
    values.to_csv('output/values_all.csv')

