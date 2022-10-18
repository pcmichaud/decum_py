#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from optim import *
from matplotlib import pyplot as plt


scenarios = ['opt','floor','nobequest']

# plot and save
table = pd.DataFrame(index=scenarios,columns=['ann','ltc','rmr'])

for scn in scenarios:
    # Load values which are compared by run_ref.py
    df = pd.read_csv('output/joint_'+scn+'.csv')
    # compute value differences for each scenarios (0 = baseline)
    table.loc[scn,'ann'] = df['buy_ann_indp'].mean()
    table.loc[scn,'ltc'] = df['buy_ltci_indp'].mean()
    table.loc[scn,'rmr'] = df['buy_rmr_indp'].mean()

print(table)

for c in table.columns:
    table[c] = table[c].astype('float64')
table.round(3).to_latex('output/optimal_why_fair.tex')





