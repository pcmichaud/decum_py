#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from optim import *
from matplotlib import pyplot as plt


scenarios = ['nomiss','transfers','nobequest','muhealth','survival','home','home_price']

# plot and save
table = pd.DataFrame(index=scenarios,columns=['ann','ltc','rmr'])

for scn in scenarios:
    # Load values which are compared by run_ref.py
    df = pd.read_csv('output/values_'+scn+'.csv')
    # compute value differences for each scenarios (0 = baseline)
    for i in range(1,13):
        df['d_value_'+str(i)] = df['value_'+str(i)] - df['value_0']
        df['buy_'+str(i)] = df['d_value_'+str(i)]>0
        if scn=='nomiss':
            print(scn,i,df['buy_'+str(i)].mean())
    table.loc[scn,'ann'] = df[['buy_'+str(i) for i in range(1,5)]].stack().mean()
    table.loc[scn,'ltc'] = df[['buy_'+str(i) for i in range(5,9)]].stack().mean()
    table.loc[scn,'rmr'] = df[['buy_'+str(i) for i in range(9,13)]].stack().mean()

print(table)

for c in table.columns:
    table[c] = table[c].astype('float64')
table.round(3).to_latex('output/optimal_why.tex')





