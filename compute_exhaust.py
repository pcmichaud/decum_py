from frame import *
import warnings
from numba import set_num_threads
import pandas as pd

if __name__ == '__main__':
	warnings.simplefilter(action='ignore')
	data = init_data()
	theta = np.load('output/estimates_reference.npy')
	values = solve_sim(data, npartitions=250,theta=theta)
	values['pexhaust85_sim'] = values['pexhaust85_sim'].astype('float64')
	values.to_csv('output/exhaust_sim.csv')
