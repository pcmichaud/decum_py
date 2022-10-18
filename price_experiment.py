from frame import *
import warnings
from numba import set_num_threads
import pandas as pd
import numpy as np

if __name__ == '__main__':
	warnings.simplefilter(action='ignore')
	loads = np.linspace(0.5,2,5)
	loads_rmr = np.linspace(-0.03,0.05,5)
	table = pd.DataFrame(index=loads,columns=['ann','ltci','rmr'])
	table_as = pd.DataFrame(index=loads,columns=['ann','ltci','rmr'])
	for load,load_rmr in zip(loads,loads_rmr):
		data = init_data()
		# load estimated parameters
		theta = np.load('output/estimates_nomiss.npy')
		# load fair prices
		prices = pd.read_csv('output/fair_prices.csv')
		prices.set_index('respid',inplace=True)
		# load other vars
		others = pd.read_csv('inputs/other_vars.csv')
		others.set_index('respid',inplace=True)
		data = data.merge(prices,left_on='respid',right_on='respid',how='left')
		data = data.merge(others,left_on='respid',right_on='respid',how='left')
		data.loc[data['price_rmr_fair'].isna(),'price_rmr_fair'] = 0.0
		# average prices per cell
		avg_prices = data.groupby(['age','female']).mean()[['price_ann_fair','price_ltci_fair','price_rmr_fair']]
		avg_prices.columns = ['price_ann_fair_avg','price_ltci_fair_avg','price_rmr_fair_avg']
		data = data.merge(avg_prices,left_on=['age','female'],right_on=['age','female'],how='left')
		print(data[['price_ann_fair_avg','price_ltci_fair_avg','price_rmr_fair_avg']].describe().transpose())
		print(data[['price_ann_fair','price_ltci_fair','price_rmr_fair']].describe().transpose())
		data['price_ann_fair'] = data['price_ann_fair']/data['price_ann_fair_avg']
		data['price_ltci_fair'] = data['price_ltci_fair']/data['price_ltci_fair_avg']
		data['price_rmr_fair'] = data['price_rmr_fair']/data['price_rmr_fair_avg']
		for c in ['price_ann_fair_avg','price_ltci_fair_avg']:
			data[c] *= load
		data['price_rmr_fair_avg'] += load_rmr
		print(data['price_rmr_fair_avg'].mean())
		values = solve_joint(data, npartitions=250,theta=theta)
		for c in ['buy_ann_joint','buy_ltci_joint','buy_rmr_joint']:
			values[c] = values[c].astype('float64')
		for c in ['buy_ann_indp','buy_ltci_indp','buy_rmr_indp']:
			values[c] = values[c].astype('float64')
		result = values[['buy_ann_indp','buy_ltci_indp','buy_rmr_indp']].mean().to_list()
		table_as.loc[load,'ann'] = values.loc[values['buy_ann_indp']==1.0,'price_ann_fair'].mean()
		table_as.loc[load,'ltci'] = values.loc[values['buy_ltci_indp']==1.0,'price_ltci_fair'].mean()
		table_as.loc[load,'rmr'] = values.loc[values['buy_rmr_indp']==1.0,'price_rmr_fair'].mean()
		table.loc[load,:] = result
		print(table)
	table.to_csv('output/price_experiment.csv')
	table_as.to_csv('output/price_experiment_as.csv')

