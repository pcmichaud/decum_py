import pandas as pd  
from frame import *
import statsmodels.api as sm

def within_difference(data):
	mean_data = data.mean(axis=1)
	for c in data.columns:
		data.loc[:,c] = data[c] - mean_data.loc[:]
	return data

def set_theta(pars):
	theta = np.zeros(pars.shape[0])
	# varepsilon
	theta[0] = np.log(pars[0])
	# d_varepsilon 
	theta[1] = np.log(pars[1])
	# gamma 
	theta[2] = np.log(pars[2])
	# d_gamma 
	theta[3] = np.log(pars[3])
	# rho 
	theta[4] = np.log(pars[4]/(1.0-pars[4]))
	# b_x 
	theta[5] = np.log(pars[5])
	# d_b_x 
	theta[6] = np.log(pars[6])
	# b_k
	theta[7] = np.log(pars[7])
	# nu_c1
	theta[8] = np.log(pars[8])
	# nu_c2 
	theta[9] = np.log(pars[9])
	# nu_h0 
	theta[10] = np.log(pars[10])
	# nu_h1 
	theta[11] = np.log(pars[11])
	# d_nu_h 
	theta[12] = np.log(pars[12])
	# miss_prob
	theta[13] = pars[13]
	theta[14] = pars[14]
	return theta 

def extract_pars(theta):
	pars = np.zeros(theta.shape[0])
	# varepsilon
	pars[0] = np.exp(theta[0])
	# d_varepsilon 
	pars[1] = np.exp(theta[1])
	# gamma 
	pars[2] = np.exp(theta[2])
	# d_gamma 
	pars[3] = np.exp(theta[3])
	# rho 
	pars[4] = np.exp(theta[4])/(1.0+np.exp(theta[4]))
	# b_x 
	pars[5] = np.exp(theta[5])
	# d_b_x 
	pars[6] = np.exp(theta[6])
	# b_k
	pars[7] = np.exp(theta[7])
	# nu_c1
	pars[8] = np.exp(theta[8])
	# nu_c2 
	pars[9] = np.exp(theta[9])
	# nu_h0 
	pars[10] = np.exp(theta[10])
	# nu_h1 
	pars[11] = np.exp(theta[11])
	# d_nu_h 
	pars[12] = np.exp(theta[12])
	# miss_prob
	pars[13] = theta[13]
	pars[14] = theta[14]
	return pars 


def concentrated_distance_within(theta, grad, data, npartitions=50):	
	# get params 
	pars = extract_pars(theta)
	# get dataset with solved expected utilities
	df = solve_df(data, npartitions=npartitions, theta=pars)
	# take difference in value with respect to baseline 
	scns = [s for s in range(1,13)]
	for s in scns:
		df['d_value_'+str(s)] = df['value_'+str(s)] - df['value_0']
	# for each set of products, take within differences
	df[['w_value_'+str(s) for s in range(1,5)]] = within_difference(df[['d_value_' 
							+str(s) for s in range(1,5)]])
	df[['w_value_'+str(s) for s in range(5,9)]] = within_difference(df[['d_value_' 
							+str(s) for s in range(5,9)]])
	df[['w_value_'+str(s) for s in range(9,13)]] = within_difference(df[['d_value_' 
							+str(s) for s in range(9,13)]])
	# take exp odds transform of probabilities
	s = 1
	for i in range(1,5):
		df['odd_'+str(s)] = np.log(df['prob_scn_ann_'+str(i)])/(1.0 - df['prob_scn_ann_'+str(i)])
		s += 1
	for i in range(1,5):
		df['odd_'+str(s)] = np.log(df['prob_scn_ltci_'+str(i)])/(1.0 - df['prob_scn_ltci_'+str(i)])
		s += 1
	for i in range(1,5):
		df['odd_'+str(s)] = np.log(df['prob_scn_rmr_'+str(i)])/(1.0 - df['prob_scn_rmr_'+str(i)])
		s += 1
	# take within deviations
	df[['w_odd_'+str(s) for s in range(1,5)]] = within_difference(df[['odd_' 
							+str(s) for s in range(1,5)]])
	df[['w_odd_'+str(s) for s in range(5,9)]] = within_difference(df[['odd_' 
							+str(s) for s in range(5,9)]])
	df[['w_odd_'+str(s) for s in range(9,13)]] = within_difference(df[['odd_' 
							+str(s) for s in range(9,13)]])
	# perform OLS to obtain estimates of sigma per product
	sigmas = [0.0,0.0,0.0]
	sum_distance = 0.0
	y = df.loc[~df['w_odd_1'].isna(),['w_odd_'+str(s) for s in range(1,5)]].stack().values
	X = df.loc[~df['w_odd_1'].isna(),['w_value_'+str(s) for s in range(1,5)]].stack().values
	model = sm.OLS(y,X,missing='drop')
	results = model.fit()
	sigmas[0] = results.params[0]
	sum_distance += results.ssr
	y = df.loc[~df['w_odd_5'].isna(),['w_odd_'+str(s) for s in range(5,9)]].stack().values
	X = df.loc[~df['w_odd_5'].isna(),['w_value_'+str(s) for s in range(5,9)]].stack().values
	model = sm.OLS(y,X,missing='drop')
	results = model.fit()
	sigmas[1] = results.params[0]
	sum_distance += results.ssr
	y = df.loc[~df['w_odd_9'].isna(),['w_odd_'+str(s) for s in range(9,12)]].stack().values
	X = df.loc[~df['w_odd_9'].isna(),['w_value_'+str(s) for s in range(9,12)]].stack().values
	model = sm.OLS(y,X,missing='drop')
	results = model.fit()
	sigmas[2] = results.params[0]
	sum_distance += results.ssr
	print('- function call summary')
	print('ssd = ', sum_distance, ' sigmas = ',sigmas)
	print('pars = ',pars)
	return sum_distance 

def concentrated_distance_levels(theta, grad, data, npartitions=50):	
	# get params 
	pars = extract_pars(theta)
	# get dataset with solved expected utilities
	df = solve_df(data, npartitions=npartitions, theta=pars)
	# take difference in value with respect to baseline 
	scns = [s for s in range(1,13)]
	for s in scns:
		df['d_value_'+str(s)] = df['value_'+str(s)] - df['value_0']
	# take exp odds transform of probabilities
	s = 1
	for i in range(1,5):
		df['odd_'+str(s)] = np.log(df['prob_scn_ann_'+str(i)])/(1.0 - df['prob_scn_ann_'+str(i)])
		s += 1
	for i in range(1,5):
		df['odd_'+str(s)] = np.log(df['prob_scn_ltci_'+str(i)])/(1.0 - df['prob_scn_ltci_'+str(i)])
		s += 1
	for i in range(1,5):
		df['odd_'+str(s)] = np.log(df['prob_scn_rmr_'+str(i)])/(1.0 - df['prob_scn_rmr_'+str(i)])
		s += 1
	# perform OLS to obtain estimates of sigma per product
	sigmas = [0.0,0.0,0.0]
	sum_distance = 0.0
	y = df.loc[~df['odd_1'].isna(),['odd_'+str(s) for s in range(1,5)]].stack().values
	X = df.loc[~df['odd_1'].isna(),['d_value_'+str(s) for s in range(1,5)]].stack().values
	X = sm.add_constant(X)
	model = sm.OLS(y,X,missing='drop')
	results = model.fit()
	sigmas[0] = results.params[1]
	sum_distance += results.ssr
	y = df.loc[~df['odd_5'].isna(),['odd_'+str(s) for s in range(5,9)]].stack().values
	X = df.loc[~df['odd_5'].isna(),['d_value_'+str(s) for s in range(5,9)]].stack().values
	X = sm.add_constant(X)
	model = sm.OLS(y,X,missing='drop')
	results = model.fit()
	sigmas[1] = results.params[1]
	sum_distance += results.ssr
	y = df.loc[~df['odd_9'].isna(),['odd_'+str(s) for s in range(9,12)]].stack().values
	X = df.loc[~df['odd_9'].isna(),['d_value_'+str(s) for s in range(9,12)]].stack().values
	X = sm.add_constant(X)
	model = sm.OLS(y,X,missing='drop')
	results = model.fit()
	sigmas[2] = results.params[1]
	sum_distance += results.ssr
	print('- function call summary')
	print('ssd = ', sum_distance, ' sigmas = ',sigmas)
	print('pars = ',pars)
	return sum_distance 



