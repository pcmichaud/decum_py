from tkinter import N
import numpy as np 
import pandas as pd 
from optim import *
from matplotlib import pyplot as plt
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

df = pd.read_csv('output/simulated_full.csv')
df['qinc'] = pd.qcut(df['totinc'],q=4)
df['qhome'] = pd.qcut(df['home_value'],q=4)
df['qwlth'] = pd.qcut(df['wealth_total'],q=4)
df['qmu'] = pd.qcut(df['mu'],q=3)
df['qxi'] = pd.qcut(df['xi'],q=3)
for a in np.arange(40):
	df['surv_'+str(a)] = np.where(~df['cons_'+str(a)].isna(),1,0)
vars_list = ['married','qinc','qhome','qwlth','qmu','qxi','pref_risk_averse','pref_live_fast','pref_home','pref_beq_money']
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
for v in vars_list:
	if v in df.columns:
		fig, ax = plt.subplots(2,2,figsize=(12, 6))
		m, low, up = smooth_profile(df[['cons_'+str(x) for x in range(cut_time)]],df[v])
		labels = df[v].value_counts().sort_index().index.to_list()
		colors = ['b','g','r','c']
		for c in range(m.shape[1]):
			ax[0,0].plot(ages,m[:,c],label=labels[c],color=colors[c])
			ax[0,0].fill_between(ages,low[:,c],up[:,c],color=colors[c],alpha=0.2)
		ax[0,0].set_title('consumption')
		m, low, up = smooth_profile(df[['wlth_'+str(x) for x in range(cut_time)]],df[v])
		labels = df[v].value_counts().sort_index().index.to_list()
		colors = ['b','g','r','c']
		for c in range(m.shape[1]):
			ax[0,1].plot(ages,m[:,c],label=labels[c],color=colors[c])
			ax[0,1].fill_between(ages,low[:,c],up[:,c],color=colors[c],alpha=0.2)
		ax[0,1].set_title('financial wealth')
		m, low, up = smooth_profile(df[['own_'+str(x) for x in range(cut_time)]],df[v])
		labels = df[v].value_counts().sort_index().index.to_list()
		colors = ['b','g','r','c']
		for c in range(m.shape[1]):
			ax[1,0].plot(ages,m[:,c],label=labels[c],color=colors[c])
			ax[1,0].fill_between(ages,low[:,c],up[:,c],color=colors[c],alpha=0.2)
		ax[1,0].set_title('own home')
		m, low, up = smooth_profile(df[['surv_'+str(x) for x in range(cut_time)]],df[v])
		labels = df[v].value_counts().sort_index().index.to_list()
		colors = ['b','g','r','c']
		for c in range(m.shape[1]):
			ax[1,1].plot(ages,m[:,c],label=labels[c],color=colors[c])
			ax[1,1].fill_between(ages,low[:,c],up[:,c],color=colors[c],alpha=0.2)
		ax[1,1].set_title('survival rate')
		fig.suptitle('Simulated profiles by '+v)
		lines_labels = fig.axes[0].get_legend_handles_labels()
		fig.legend(lines_labels[0], lines_labels[1],ncol=4,loc="lower right")
		for a in ax.flat:
			a.set(xlabel='time', ylabel='mean - rate')
		plt.tight_layout() 
		plt.savefig('output/simulated_path_ref_'+v+'.png',dpi=1200)
		

