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
plt.figure()
df[['buy_'+str(i) for i in range(1,13)]].mean().plot.bar()
plt.savefig('buy_ref.png',dpi=1200)

