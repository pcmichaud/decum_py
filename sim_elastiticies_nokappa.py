#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from optim import *
from matplotlib import pyplot as plt
from itertools import product
from linearmodels import PanelOLS
import statsmodels.api as sm
from itertools import product 

# Load values which are compared by run_ref.py
df = pd.read_csv('output/values_nokappa_with_deltas.csv')
pd.set_option('display.max_rows', 500)
# compute value differences for each scenarios (0 = baseline)
for i in range(1,13):
	df['d_value_'+str(i)] = df['value_'+str(i)] - df['value_0']

# compute indicator for whether value diff is positive (buy)
for i in range(1,13):
	df['buy_'+str(i)] = np.where(df['d_value_'+str(i)]>0,1.0,0.0)

# compute predicted probabilities
for i in range(1,5):
    df['prob_'+str(i)] = np.exp(-df['delta_ann'] + df['sigma_ann']*df['d_value_'+str(i)])
    df['prob_'+str(i)] = df['prob_'+str(i)]/(1.0 + df['prob_'+str(i)])

for i in range(5,9):
    df['prob_'+str(i)] = np.exp(-df['delta_ltc'] + df['sigma_ltc']*df['d_value_'+str(i)])
    df['prob_'+str(i)] = df['prob_'+str(i)]/(1.0 + df['prob_'+str(i)])

for i in range(9,13):
    df['prob_'+str(i)] = np.exp(-df['delta_rmr'] + df['sigma_rmr']*df['d_value_'+str(i)])
    df['prob_'+str(i)] = df['prob_'+str(i)]/(1.0 + df['prob_'+str(i)])

products = ['ann','ltc','rmr']
elas = ['price','benefit']
lists = [elas,products]
tuples = list(product(*lists))

table = pd.DataFrame(index=pd.MultiIndex.from_tuples(tuples),columns=['predicted','data','model'])


# annuities
keep_vars = ['respid']
for i in range(1,5):
    keep_vars.append('prob_'+str(i))
for i in range(1,5):
    keep_vars.append('buy_'+str(i))
for i in range(1,5):
    keep_vars.append('prob_scn_ann_'+str(i))
for i in range(1,5):
    keep_vars.append('prem_scn_ann_'+str(i))
for i in range(1,5):
    keep_vars.append('ben_scn_ann_'+str(i))
df_p = df.loc[:,keep_vars]

df_p = pd.wide_to_long(df_p,['prob_','buy_','prob_scn_ann_','prem_scn_ann_','ben_scn_ann_'],i='respid',j='scn')
df_p.columns = ['psim','pmodel','pdata','price','benfs']
for c in df_p.columns:
    df_p[c] = np.where(df_p[c]==-999,np.nan,df_p[c])
df_p.dropna(inplace=True)
print(df_p.describe())
y = df_p.loc[:,'psim']
X = sm.add_constant(df_p.loc[:,['benfs','price']])
mod = PanelOLS(y,X,entity_effects=True)
results = mod.fit()
print(results)
print('annuities p  (sim) = ',results.params[2]*X['price'].mean()/y.mean())
print('annuities b  (sim) = ',results.params[1]*X['benfs'].mean()/y.mean())
table.loc[('price','ann'),'predicted'] = results.params[2]*X['price'].mean()/y.mean()
table.loc[('benefit','ann'),'predicted'] = results.params[1]*X['benfs'].mean()/y.mean()



y = df_p.loc[:,'pdata']
X = sm.add_constant(df_p.loc[:,['benfs','price']])
mod = PanelOLS(y,X,entity_effects=True)
results = mod.fit()
print(results)
print('annuities p (data) = ',results.params[2]*X['price'].mean()/y.mean())
print('annuities b  (data) = ',results.params[1]*X['benfs'].mean()/y.mean())
table.loc[('price','ann'),'data'] = results.params[2]*X['price'].mean()/y.mean()
table.loc[('benefit','ann'),'data'] = results.params[1]*X['benfs'].mean()/y.mean()


y = df_p.loc[:,'pmodel']
X = sm.add_constant(df_p.loc[:,['benfs','price']])
mod = PanelOLS(y,X,entity_effects=True)
results = mod.fit()
print(results)
print('annuities p (model) = ',results.params[2]*X['price'].mean()/y.mean())
print('annuities b  (model) = ',results.params[1]*X['benfs'].mean()/y.mean())
table.loc[('price','ann'),'model'] = results.params[2]*X['price'].mean()/y.mean()
table.loc[('benefit','ann'),'model'] = results.params[1]*X['benfs'].mean()/y.mean()

# ltci
keep_vars = ['respid']
for i in range(5,9):
    keep_vars.append('prob_'+str(i))
for i in range(5,9):
    keep_vars.append('buy_'+str(i))
for i in range(1,5):
    keep_vars.append('prob_scn_ltci_'+str(i))
for i in range(1,5):
    keep_vars.append('prem_scn_ltci_'+str(i))
for i in range(1,5):
    keep_vars.append('ben_scn_ltci_'+str(i))
df_p = df.loc[:,keep_vars]
for i in range(5,9):
    df_p = df_p.rename({'prob_'+str(i):'prob_'+str(i-4)},axis=1)
for i in range(5,9):
    df_p = df_p.rename({'buy_'+str(i):'buy_'+str(i-4)},axis=1)
df_p = pd.wide_to_long(df_p,['prob_','buy_','prob_scn_ltci_','prem_scn_ltci_','ben_scn_ltci_'],i='respid',j='scn')
print(df_p.describe())
df_p.columns = ['psim','pmodel','pdata','price','benfs']
for c in df_p.columns:
    df_p[c] = np.where(df_p[c]==-999,np.nan,df_p[c])
df_p.dropna(inplace=True)
print(df_p.describe())
y = df_p.loc[:,'psim']
X = sm.add_constant(df_p.loc[:,['benfs','price']])
mod = PanelOLS(y,X,entity_effects=True)
results = mod.fit()
print('ltci p (sim) = ',results.params[2]*X['price'].mean()/y.mean())
print('ltci b  (sim) = ',results.params[1]*X['benfs'].mean()/y.mean())
table.loc[('price','ltc'),'predicted'] = results.params[2]*X['price'].mean()/y.mean()
table.loc[('benefit','ltc'),'predicted'] = results.params[1]*X['benfs'].mean()/y.mean()

y = df_p.loc[:,'pdata']
X = sm.add_constant(df_p.loc[:,['benfs','price']])
mod = PanelOLS(y,X,entity_effects=True)
results = mod.fit()
print('ltci p (data) = ',results.params[2]*X['price'].mean()/y.mean())
print('ltci b  (data) = ',results.params[1]*X['benfs'].mean()/y.mean())
table.loc[('price','ltc'),'data'] = results.params[2]*X['price'].mean()/y.mean()
table.loc[('benefit','ltc'),'data'] = results.params[1]*X['benfs'].mean()/y.mean()

y = df_p.loc[:,'pmodel']
X = sm.add_constant(df_p.loc[:,['benfs','price']])
mod = PanelOLS(y,X,entity_effects=True)
results = mod.fit()
print('ltci p (model) = ',results.params[2]*X['price'].mean()/y.mean())
print('ltci b  (model) = ',results.params[1]*X['benfs'].mean()/y.mean())
table.loc[('price','ltc'),'model'] = results.params[2]*X['price'].mean()/y.mean()
table.loc[('benefit','ltc'),'model'] = results.params[1]*X['benfs'].mean()/y.mean()

# rmr
keep_vars = ['respid']
for i in range(9,13):
    keep_vars.append('prob_'+str(i))
for i in range(9,13):
    keep_vars.append('buy_'+str(i))
for i in range(1,5):
    keep_vars.append('prob_scn_rmr_'+str(i))
for i in range(1,5):
    keep_vars.append('int_scn_rmr_'+str(i))
for i in range(1,5):
    keep_vars.append('loan_scn_rmr_'+str(i))
df_p = df.loc[:,keep_vars]
for i in range(9,13):
    df_p = df_p.rename({'prob_'+str(i):'prob_'+str(i-8)},axis=1)
for i in range(9,13):
    df_p = df_p.rename({'buy_'+str(i):'buy_'+str(i-8)},axis=1)
df_p = pd.wide_to_long(df_p,['prob_','buy_','prob_scn_rmr_','int_scn_rmr_','loan_scn_rmr_'],i='respid',j='scn')
df_p.columns = ['psim','pmodel','pdata','price','benfs']
for c in df_p.columns:
    df_p[c] = np.where(df_p[c]==-999,np.nan,df_p[c])
df_p.dropna(inplace=True)
print(df_p.describe())
y = df_p.loc[:,'psim']
X = sm.add_constant(df_p.loc[:,['benfs','price']])
mod = PanelOLS(y,X,entity_effects=True)
results = mod.fit()
print(results)
print('rmr p (sim) = ',results.params[2]*X['price'].mean()/y.mean())
print('rmr b  (sim) = ',results.params[1]*X['benfs'].mean()/y.mean())
table.loc[('price','rmr'),'predicted'] = results.params[2]*X['price'].mean()/y.mean()
table.loc[('benefit','rmr'),'predicted'] = results.params[1]*X['benfs'].mean()/y.mean()


y = df_p.loc[:,'pdata']
X = sm.add_constant(df_p.loc[:,['benfs','price']])
mod = PanelOLS(y,X,entity_effects=True)
results = mod.fit()
print(results)
print('rmr p (data) = ',results.params[2]*X['price'].mean()/y.mean())
print('rmr b  (data) = ',results.params[1]*X['benfs'].mean()/y.mean())
table.loc[('price','rmr'),'data'] = results.params[2]*X['price'].mean()/y.mean()
table.loc[('benefit','rmr'),'data'] = results.params[1]*X['benfs'].mean()/y.mean()


y = df_p.loc[:,'pmodel']
X = sm.add_constant(df_p.loc[:,['benfs','price']])
mod = PanelOLS(y,X,entity_effects=True)
results = mod.fit()
print(results)
print('rmr p (model) = ',results.params[2]*X['price'].mean()/y.mean())
print('rmr b  (model) = ',results.params[1]*X['benfs'].mean()/y.mean())
table.loc[('price','rmr'),'model'] = results.params[2]*X['price'].mean()/y.mean()
table.loc[('benefit','rmr'),'model'] = results.params[1]*X['benfs'].mean()/y.mean()

for c in table.columns:
    table[c] = table[c].astype('float64')
table = table.round(3)
print(table)
table.to_latex('output/table_simulated_elasticities_nokappa.tex')