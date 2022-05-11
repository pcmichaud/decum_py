from tkinter import N
import numpy as np 
import pandas as pd 
from optim import *
from matplotlib import pyplot as plt
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

dfs = []

scns = ['nokappa','nobequest','transfers','muhealth','medrisk']
for scn in scns:
	df = pd.read_csv('output/simulated_'+scn+'.csv')
	df['qinc'] = pd.qcut(df['totinc'],q=3)
	for a in np.arange(40):
		df['surv_'+str(a)] = np.where(~df['cons_'+str(a)].isna(),1,0)
		df['totwlth_'+str(a)] = df['wlth_'+str(a)] + df['home_'+str(a)]
	dfs.append(df)
vars_list = ['married']
scns = ['baseline','no bequest','low transfers','state-independent $\\nu_c$','no med risk']
cut_time = 30

def smooth_profile(data,byvar):
	mean = data.groupby(byvar).mean().transpose().to_numpy()
	se   = data.groupby(byvar).std().transpose().to_numpy()
	n =  data.groupby(byvar).count().transpose().to_numpy()
	n_cells = mean.shape[0]
	ages = np.arange(n_cells)
	for c in range(se.shape[1]):
		se[:,c] = se[:,c]/np.sqrt(n[:,c])
		res = lowess(mean[:,c], ages, missing='drop')
		mean[:,c] = res[:,1]
	return mean, mean - 1.96*se, mean + 1.96*se
ages =  np.arange(cut_time)

# for singles
colors = ['b','g','r','c','m']
plt.figure()
for i,scn in enumerate(scns):
	df = dfs[i]
	d_m, d_low, d_up = smooth_profile(df[['wlth_'+str(x) for x in range(cut_time)]],df['married'])
	if scn=='baseline':
		plt.plot(ages,d_m[:,0],color=colors[i],label=scn)
	else :
		plt.plot(ages,d_m[:,0],color=colors[i],linestyle='dashed',label=scn)
plt.xlabel('age')
plt.ylabel('financial wealth')
plt.legend()
plt.show()
plt.savefig('output/simulated_path_why_finwlth_singles.eps',format='eps')
		

colors = ['b','g','r','c','m']
plt.figure()
for i,scn in enumerate(scns):
	df = dfs[i]
	d_m, d_low, d_up = smooth_profile(df[['wlth_'+str(x) for x in range(cut_time)]],df['married'])
	if scn=='baseline':
		plt.plot(ages,d_m[:,1],color=colors[i],label=scn)
	else :
		plt.plot(ages,d_m[:,1],color=colors[i],linestyle='dashed',label=scn)
plt.xlabel('age')
plt.ylabel('financial wealth')
plt.legend()
plt.show()
plt.savefig('output/simulated_path_why_finwlth_couples.eps',format='eps')


# for singles
colors = ['b','g','r','c','m']
plt.figure()
for i,scn in enumerate(scns):
	df = dfs[i]
	d_m, d_low, d_up = smooth_profile(df[['totwlth_'+str(x) for x in range(cut_time)]],df['married'])
	if scn=='baseline':
		plt.plot(ages,d_m[:,0],color=colors[i],label=scn)
	else :
		plt.plot(ages,d_m[:,0],color=colors[i],linestyle='dashed',label=scn)
plt.xlabel('age')
plt.ylabel('total wealth')
plt.legend()
plt.show()
plt.savefig('output/simulated_path_why_totwlth_singles.eps',format='eps')
		

colors = ['b','g','r','c','m']
plt.figure()
for i,scn in enumerate(scns):
	df = dfs[i]
	d_m, d_low, d_up = smooth_profile(df[['totwlth_'+str(x) for x in range(cut_time)]],df['married'])
	if scn=='baseline':
		plt.plot(ages,d_m[:,1],color=colors[i],label=scn)
	else :
		plt.plot(ages,d_m[:,1],color=colors[i],linestyle='dashed',label=scn)
plt.xlabel('age')
plt.ylabel('total wealth')
plt.legend()
plt.show()
plt.savefig('output/simulated_path_why_totwlth_couples.eps',format='eps')
		