from frame import *
import warnings
from numba import set_num_threads
import pandas as pd

# annuity experiment
if __name__ == '__main__':
    warnings.simplefilter(action='ignore')
    loads = [0.25,0.5,0.75,1.0,1.25,1.50,1.75,2.0]
    for load in loads:
        print('load = ',load)
        data = init_data()
        for scn in [1,2,3,4]:
            data['ben_scn_ann_'+str(scn)] *= load 
        for scn in [1,2,3,4]:
            data['prem_scn_ltci_'+str(scn)] *= load 
        for scn in [1,2,3,4]:
            data['int_scn_rmr_'+str(scn)] *= load 
        theta = np.load('output/estimates_nomiss.npy')
        values = solve_df(data, npartitions=250,theta=theta)
        print(values[['value_'+str(x) for x in range(13)]].describe().transpose())
        values.to_csv('output/values_loads_'+str(int(load*100))+'.csv')

