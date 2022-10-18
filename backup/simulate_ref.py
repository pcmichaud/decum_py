from frame import *
import warnings
from numba import set_num_threads
import pandas as pd

if __name__ == '__main__':
    warnings.simplefilter(action='ignore')
    data = init_data()
    #data = data.sample(n=500,random_state=1234)
    values = simulate_df(data, npartitions=250,theta=None)
    print(values[['cons_'+str(x) for x in range(45)]].describe().transpose())
    print(values[['own_'+str(x) for x in range(45)]].describe().transpose())
    print(values[['wlth_'+str(x) for x in range(45)]].describe().transpose())
    values.to_csv('output/simulated_full.csv')

