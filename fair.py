from frame import *
import warnings
from numba import set_num_threads
import pandas as pd

if __name__ == '__main__':
	warnings.simplefilter(action='ignore')
	data = init_data()
	theta = np.load('output/estimates_reference.npy')
	values = solve_fair(data, npartitions=250,theta=theta)
	for c in ['price_ann_fair','price_ltci_fair','price_rmr_fair']:
		values[c] = values[c].astype('float64')
	print(values[['price_ann_fair','price_ltci_fair','price_rmr_fair']].describe().transpose())
	values = values[['price_ann_fair','price_ltci_fair','price_rmr_fair']]
	values.to_csv('output/fair_prices.csv')
