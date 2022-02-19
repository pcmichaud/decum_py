from frame import *
import warnings
from numba import set_num_threads

if __name__ == '__main__':

    warnings.simplefilter(action='ignore')
    #set_num_threads(2)
    values = solve_df()
    print(values[['value_'+str(x) for x in range(13)]].describe().transpose())

