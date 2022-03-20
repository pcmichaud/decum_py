#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from optim import *
from matplotlib import pyplot as plt


# Load values which are compared by run_ref.py
df = pd.read_csv('output/values_full.csv')
pd.set_option('display.max_rows', 500)
# compute value differences for each scenarios (0 = baseline)
for i in range(1,13):
	df['d_value_'+str(i)] = df['value_'+str(i)] - df['value_0']
print(df[['d_value_'+str(i) for i in range(1,13)]].describe().transpose())

# compute indicator for whether value diff is positive (buy)
for i in range(1,13):
	df['buy_'+str(i)] = df['d_value_'+str(i)]>0

# plot and save
labels = ['prob_scn_ann_'+str(i) for i in range(1,5)]
for i in range(1,5):
    labels.append('prob_scn_ltci_'+str(i))
for i in range(1,5):
    labels.append('prob_scn_rmr_'+str(i))
mod_buy = df[['buy_'+str(i) for i in range(1,13)]].mean()
dat_buy = df[labels].mean()

labels_tab= []
for i in range(1,5):
    labels_tab.append('annuities ('+str(i)+')')
for i in range(1,5):
    labels_tab.append('ltci ('+str(i)+')')
for i in range(1,5):
    labels_tab.append('reverse mort. ('+str(i)+')')

table = pd.DataFrame(index=labels_tab,columns=['data','model (optimal)'])
table['data'] = dat_buy.to_list()
table['model (optimal)'] = mod_buy.to_list()
print(table)
table.round(3).to_latex('output/stats_values_ref.tex')

