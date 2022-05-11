import pandas as pd  
from frame import *
import statsmodels.api as sm

def within_difference(data):
	mean_data = data.mean(axis=1)
	for c in data.columns:
		data.loc[:,c] = data[c] - mean_data.loc[:]
	return data

def set_theta(pars, isfree):
	theta = np.zeros(int(np.sum(isfree)))
	i = 0
	# varepsilon
	if isfree[0]==1:
		theta[i] = np.log(pars[0])
		i += 1
	# d_varepsilon 
	if isfree[1]==1:
		theta[i] = np.log(pars[1])
		i += 1
	# gamma 
	if isfree[2]==1:
		theta[i] = np.log(pars[2])
		i +=1
	# d_gamma 
	if isfree[3]==1:
		theta[i] = np.log(pars[3])
		i +=1
	# rho 
	if isfree[4]==1:
		theta[i] = np.log(pars[4]/(1.0-pars[4]))
		i +=1 
	# b_x 
	if isfree[5]==1:
		theta[i] = np.log(pars[5])
		i +=1 
	# d_b_x 
	if isfree[6]==1:
		theta[i] = np.log(pars[6])
		i +=1 
	# b_k
	if isfree[7]==1:
		theta[i] = np.log(pars[7])
		i +=1 
	# nu_c1
	if isfree[8]==1:
		theta[i] = np.log(pars[8])
		i +=1
	# nu_c2 
	if isfree[9]==1:
		theta[i] = np.log(pars[9])
		i +=1
	# nu_h
	if isfree[10]==1:
		theta[i] = np.log(pars[10])
		i +=1
	# d_nu_h 
	if isfree[11]==1:
		theta[i] = np.log(pars[11])
		i +=1
	# miss_prob
	if isfree[12]==1:
		theta[i] = pars[12]
		i +=1 
	if isfree[13]==1:
		theta[i] = pars[13]
		i +=1 
	return theta 

def extract_pars(theta, isfree, ipars):
	pars = np.zeros(ipars.shape[0])
	i = 0
	# varepsilon
	if isfree[0]==1:
		pars[0] = np.exp(theta[i])
		i +=1
	else :
		pars[0] = ipars[0]
	# d_varepsilon 
	if isfree[1]==1:
		pars[1] = np.exp(theta[i])
		i +=1
	else :
		pars[1] = ipars[1]
	# gamma 
	if isfree[2]==1:
		pars[2] = np.exp(theta[i])
		i +=1 
	else :
		pars[2] = ipars[2]
	# d_gamma 
	if isfree[3]==1:
		pars[3] = np.exp(theta[i])
		i +=1 
	else :
		pars[3] = ipars[3]
	# rho 
	if isfree[4]==1:
		pars[4] = np.exp(theta[i])/(1.0+np.exp(theta[i]))
		i +=1
	else :
		pars[4] = ipars[4]
	# b_x 
	if isfree[5]==1:
		pars[5] = np.exp(theta[i])
		i +=1
	else :
		pars[5] = ipars[5]
	# d_b_x 
	if isfree[6]==1:
		pars[6] = np.exp(theta[i])
		i +=1 
	else :
		pars[6] = ipars[6]
	# b_k
	if isfree[7]==1:
		pars[7] = np.exp(theta[i])
		i +=1 
	else :
		pars[7] = ipars[7]
	# nu_c1
	if isfree[8]==1:
		pars[8] = np.exp(theta[i])
		i +=1
	else :
		pars[8] = ipars[8]
	# nu_c2 
	if isfree[9]==1:
		pars[9] = np.exp(theta[i])
		i +=1 
	else :
		pars[9] = ipars[9]
	# nu_h 
	if isfree[10]==1:
		pars[10] = np.exp(theta[i])
		i +=1 
	else :
		pars[10] = ipars[10]
	# d_nu_h 
	if isfree[11]==1:
		pars[11] = np.exp(theta[i])
		i +=1 
	else :
		pars[11] = ipars[11]
	# miss_prob
	if isfree[12]==1:
		pars[12] = theta[i]
		i +=1 
	else :
		pars[12] = ipars[12]
	if isfree[13]==1:
		pars[13] = theta[i]
		i +=1 
	else :
		pars[13] = ipars[13]
	return pars 


def concentrated_distance_within(theta, grad, data, isfree, ipars, scn_name, npartitions=50):	
	# get params 
	pars = extract_pars(theta,isfree,ipars)
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
	sigmas = np.zeros((3,2))
	sum_distance = 0.0
	for k in [0,1]:
		cond = (~df['w_odd_1'].isna()) & (df['know_ann']==k)
		y = df.loc[cond,['w_odd_'+str(s) for s in range(1,5)]].stack().values
		X = df.loc[cond,['w_value_'+str(s) for s in range(1,5)]].stack().values
		model = sm.OLS(y,X,missing='drop')
		results = model.fit()
		sigmas[0,k] = results.params[0]
		sum_distance += results.ssr
	for k in [0,1]:
		cond = (~df['w_odd_5'].isna()) & (df['know_ltci']==k)
		y = df.loc[cond,['w_odd_'+str(s) for s in range(5,9)]].stack().values
		X = df.loc[cond,['w_value_'+str(s) for s in range(5,9)]].stack().values
		model = sm.OLS(y,X,missing='drop')
		results = model.fit()
		sigmas[1,k] = results.params[0]
		sum_distance += results.ssr
	for k in [0,1]:
		cond = (~df['w_odd_9'].isna()) & (df['know_rmr']==k)
		y = df.loc[cond,['w_odd_'+str(s) for s in range(9,12)]].stack().values
		X = df.loc[cond,['w_value_'+str(s) for s in range(9,12)]].stack().values
		model = sm.OLS(y,X,missing='drop')
		results = model.fit()
		sigmas[2,k] = results.params[0]
		sum_distance += results.ssr
	print('- function call summary')
	print('ssd = ', sum_distance, ' sigmas = ',sigmas)
	print('pars = ',pars)
	np.save('output/sigmas_'+scn_name,sigmas)
	return sum_distance 

def residuals_within(theta, sigmas, data, isfree, ipars, npartitions=50):	
	# get params 
	pars = extract_pars(theta,isfree,ipars)
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
	residuals = pd.DataFrame(index=df.index,columns=[x for x in range(1,13)])
	df['sigma_ann'] = np.where(df['know_ann']==1,sigmas[0,1],sigmas[0,0])
	df['sigma_ltc'] = np.where(df['know_ltci']==1,sigmas[1,1],sigmas[1,0])
	df['sigma_rmr'] = np.where(df['know_rmr']==1,sigmas[2,1],sigmas[2,0])
	for s in range(1,5):
		residuals.loc[:,s] = df.loc[:,'w_odd_'+str(s)] - df['sigma_ann']* df.loc[:,'w_value_'+str(s)]
	for s in range(5,9):
		residuals.loc[:,s] = df.loc[:,'w_odd_'+str(s)] - df['sigma_ltc']* df.loc[:,'w_value_'+str(s)]
	for s in range(9,13):
		residuals.loc[:,s] = df.loc[:,'w_odd_'+str(s)] - df['sigma_rmr']* df.loc[:,'w_value_'+str(s)]
	return residuals

def g_within(theta, sigmas, data, isfree, ipars, npartitions=50):	
	# get params 
	pars = extract_pars(theta,isfree,ipars)
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
	gs = pd.DataFrame(index=df.index,columns=[x for x in range(1,13)])
	df['sigma_ann'] = np.where(df['know_ann']==1,sigmas[0,1],sigmas[0,0])
	df['sigma_ltc'] = np.where(df['know_ltci']==1,sigmas[1,1],sigmas[1,0])
	df['sigma_rmr'] = np.where(df['know_rmr']==1,sigmas[2,1],sigmas[2,0])
	for s in range(1,5):
		gs.loc[:,s] = df['sigma_ann'] * df.loc[:,'w_value_'+str(s)]
	for s in range(5,9):
		gs.loc[:,s] = df['sigma_ltc']* df.loc[:,'w_value_'+str(s)]
	for s in range(9,13):
		gs.loc[:,s] = df['sigma_rmr'] * df.loc[:,'w_value_'+str(s)]
	return gs
