#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from optim import *
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind 


# Load values which are compared by run_ref.py
df = pd.read_csv('output/values_full.csv')
pd.set_option('display.max_rows', 500)
# compute value differences for each scenarios (0 = baseline)
for i in range(1,13):
	df['d_value_'+str(i)] = df['value_'+str(i)] - df['value_0']
values = df.loc[:,['d_value_'+str(i) for i in range(1,13)]]
labs = []
for i in range(1,5):
    labs.append('ann_'+str(i))
for i in range(1,5):
    labs.append('ltc_'+str(i))
for i in range(1,5):
    labs.append('rmr_'+str(i))
values.columns = labs
print(values.describe().transpose())

# get probabilities
labs_prob = ['prob_scn_ann_'+str(i) for i in range(1,5)]
for i in range(1,5):
    labs_prob.append('prob_scn_ltci_'+str(i))
for i in range(1,5):
    labs_prob.append('prob_scn_rmr_'+str(i))
probs = df.loc[:,labs_prob]
probs.columns = labs
print(probs.describe().transpose())
# compute log odds
for c in probs.columns:
    probs[c] = np.log(probs[c]/(1.0-probs[c]))

# sigmas
sigmas = np.load('output/sigmas_ref.npy') 

df['sigma_ann'] = np.where(df['know_ann']==1,sigmas[0,1],sigmas[0,0])
df['sigma_ltc'] = np.where(df['know_ltci']==1,sigmas[1,1],sigmas[1,0])
df['sigma_rmr'] = np.where(df['know_rmr']==1,sigmas[2,1],sigmas[2,0])


# deltas 
deltas = pd.DataFrame(index=probs.index,columns=['ann','ltc','rmr'])
j = 0
for p in ['ann','ltc','rmr']:
    labs = [p+'_'+str(i) for i in range(1,5)]
    deltas['delta_'+p] = -(probs[labs].mean(axis=1) - df['sigma_'+p]*values[labs].mean(axis=1))
    j +=1

print(deltas.describe().transpose())

# merge back
df = df.merge(deltas,left_index=True,right_index=True,how='left')
df.to_csv('output/values_full_with_deltas.csv')

def ecdf(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]


# figures 
plt.figure()
colors = ['mediumseagreen','indianred','cornflowerblue']
for i,p in enumerate(['ann','ltc','rmr']):
    x, y = ecdf(deltas['delta_'+p])
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.)
    plt.plot(x, y, drawstyle='steps-post',label=p,color=colors[i])
plt.legend()
plt.ylabel('cdf')
plt.xlabel('$\\delta$')
plt.savefig('output/deltas_full.eps',format='eps')
plt.show()

# stats by characteristics
deltas = ['delta_'+p for p in ['ann','ltc','rmr']]

print(df.columns.to_list())

ann = df.groupby('know_ann').mean()['delta_ann']
ltc = df.groupby('know_ltci').mean()['delta_ltc']
rmr = df.groupby('know_rmr').mean()['delta_rmr']

table = pd.DataFrame(index=['annuities','ltci','reverse mort.'],columns=['does not know','knows'],dtype=np.float64)

table.loc['annuities',:] = ann.to_list()
table.loc['ltci',:] = ltc.to_list()
table.loc['reverse mort.',:] = rmr.to_list()

print(df['know_ann'].value_counts())
test_stats = []
test_stat = ttest_ind(df.loc[df['know_ann']==1,'delta_ann'], df.loc[df['know_ann']==0,'delta_ann'],nan_policy='omit')
test_stats.append(test_stat[0])
test_stat = ttest_ind(df.loc[df['know_ltci']==1,'delta_ltc'], df.loc[df['know_ltci']==0,'delta_ltc'],nan_policy='omit')
test_stats.append(test_stat[0])
test_stat = ttest_ind(df.loc[df['know_rmr']==1,'delta_rmr'], df.loc[df['know_rmr']==0,'delta_rmr'],nan_policy='omit')
test_stats.append(test_stat[0])
table['t-value'] = test_stats
print(table)
table.round(3).to_latex('output/deltas_by_product.tex')
