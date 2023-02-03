from frame import *
import warnings
from numba import set_num_threads
import pandas as pd
from matplotlib import pyplot as plt

warnings.simplefilter(action='ignore')

data = pd.read_csv('output/exhaust_sim.csv')
data.set_index('respid',inplace=True)
other = pd.read_csv('inputs/other_vars.csv')
other.set_index('respid',inplace=True)
data = data.merge(other,left_index=True,right_index=True)
data = data.loc[data.pexhaust.isna()==False,:]
print(data[['pexhaust85_sim','pexhaust']].describe())

print(data[['pexhaust85_sim','pexhaust']].corr())

data.to_csv('output/data_with_exhaust_sim.csv')



