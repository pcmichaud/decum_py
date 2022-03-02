from frame import *
import warnings
from numba import set_num_threads
import pandas as pd 

if __name__ == '__main__':
    warnings.simplefilter(action='ignore')
    data = init_data()
    #data = data.iloc[:500,:]
    values = simulate_df(data, npartitions=250,theta=None)
    print(values[['cons_'+str(x) for x in range(40)]].describe().transpose())
    print(values[['own_'+str(x) for x in range(40)]].describe().transpose())
    print(values[['wlth_'+str(x) for x in range(40)]].describe().transpose())
    values.to_csv('output/simulated_ref.csv')
