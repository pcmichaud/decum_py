from frame import *
import warnings
from numba import set_num_threads
import pandas as pd

if __name__ == '__main__':
    warnings.simplefilter(action='ignore')
    data = init_data()
    theta = np.load('output/estimates_nokappa.npy')
    theta[0] = 1.25
    theta[1] = 0.0
    theta[2] = 0.5
    theta[3] = 0.0
    theta[4] = 0.5
    theta[5] = 0.0
    theta[6] = 0.0
    theta[7] = 8.5
    theta[8] = 1.0
    theta[9] = 1.0
    theta[10] = 1.0
    theta[11] = 0.0
    theta[12] = 0.0
    theta[13] = 0.0
    values = simulate_df(data, npartitions=250,theta=theta)
    print(values[['cons_'+str(x) for x in range(45)]].describe().transpose())
    print(values[['own_'+str(x) for x in range(45)]].describe().transpose())
    print(values[['wlth_'+str(x) for x in range(45)]].describe().transpose())
    values.to_csv('output/simulated_bo.csv')

