#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from optim import *
from matplotlib import pyplot as plt


# Load values which are compared by run_ref.py
df = pd.read_csv('output/values_bo_with_deltas.csv')
pd.set_option('display.max_rows', 500)
# compute value differences for each scenarios (0 = baseline)
for i in range(1,13):
	df['d_value_'+str(i)] = df['value_'+str(i)] - df['value_0']
print(df[['d_value_'+str(i) for i in range(1,13)]].describe().transpose())

# compute indicator for whether value diff is positive (buy)
for i in range(1,13):
	df['buy_'+str(i)] = df['d_value_'+str(i)]>0

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


# plot and save
table = pd.DataFrame(index=['ann','ltc','rmr'],columns=['data','predicted','model (optimal)'])

table.loc['ann','data'] = df[['prob_scn_ann_'+str(i) for i in range(1,5)]].stack().mean()
table.loc['ltc','data'] = df[['prob_scn_ltci_'+str(i) for i in range(1,5)]].stack().mean()
table.loc['rmr','data'] = df[['prob_scn_rmr_'+str(i) for i in range(1,5)]].stack().mean()

table.loc['ann','model (optimal)'] = df[['buy_'+str(i) for i in range(1,5)]].stack().mean()
table.loc['ltc','model (optimal)'] = df[['buy_'+str(i) for i in range(5,9)]].stack().mean()
table.loc['rmr','model (optimal)'] = df[['buy_'+str(i) for i in range(9,13)]].stack().mean()

table.loc['ann','predicted'] = df[['prob_'+str(i) for i in range(1,5)]].stack().mean()
table.loc['ltc','predicted'] = df[['prob_'+str(i) for i in range(5,9)]].stack().mean()
table.loc['rmr','predicted'] = df[['prob_'+str(i) for i in range(9,13)]].stack().mean()

print(table)

for c in table.columns:
    table[c] = table[c].astype('float64')
table.round(3).to_latex('output/stats_values_bo.tex')

