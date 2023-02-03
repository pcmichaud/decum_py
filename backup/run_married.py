from frame import *
import warnings
from numba import set_num_threads
import pandas as pd

if __name__ == '__main__':
    warnings.simplefilter(action='ignore')
    data = init_data()
    data['married'] = 0
    theta = np.load('output/estimates_reference.npy')
    values = solve_df(data, npartitions=250,theta=theta)
    print(values[['value_'+str(x) for x in range(13)]].describe().transpose())
    values.to_csv('output/values_reference.csv')

