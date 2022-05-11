from frame import *
import warnings
from numba import set_num_threads
import pandas as pd

if __name__ == '__main__':
    warnings.simplefilter(action='ignore')
    data = init_data()
    # set floor manually to 50% of its value (in budget.py)
    theta = np.load('output/estimates_nokappa.npy')
    values = simulate_df(data, npartitions=250,theta=theta)
    print(values[['cons_'+str(x) for x in range(45)]].describe().transpose())
    print(values[['own_'+str(x) for x in range(45)]].describe().transpose())
    print(values[['wlth_'+str(x) for x in range(45)]].describe().transpose())
    values.to_csv('output/simulated_transfers.csv')

