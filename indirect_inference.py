#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from optim import *
from matplotlib import pyplot as plt
import statsmodels.api as sm

# Load values which are compared by run_ref.py
df = pd.read_csv('output/values_full_with_deltas.csv')
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




pr_s = df[['prob_'+str(i) for i in range(1,5)]].stack()
pr_d = df[['prob_scn_ann_'+str(i) for i in range(1,5)]].stack()

# get prices

price = df[['prem_scn_ann_'+str(i) for i in range(1,5)]].stack()
benfs = df[['ben_scn_ann'+str(i) for i in range(1,5)]].stack()




