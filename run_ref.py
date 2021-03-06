from frame import *
import warnings
from numba import set_num_threads
import pandas as pd

if __name__ == '__main__':
    warnings.simplefilter(action='ignore')
    data = init_data()
   

#    data = data.sample(n=500,random_state=1234)
    values = solve_df(data, npartitions=250,theta=None)
    print(values[['value_'+str(x) for x in range(13)]].describe().transpose())
    values.to_csv('output/values_full.csv')

