#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from optim import *
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

# Load values which are compared by run_ref.py
df = pd.read_csv('output/values_ez_with_deltas.csv')
pd.set_option('display.max_rows', 500)
print(df.columns.to_list())
others = pd.read_csv('inputs/other_vars.csv')
df = df.merge(others,left_on='respid',right_on='respid',how='left')
df.rename({'know_ltci':'know_ltc'},axis=1,inplace=True)
df['know'] = df[['know_ann','know_ltc','know_rmr']].sum(axis=1)
results = []
for p in ['ann','ltc','rmr']:
    Xs = df[['age','female','married','college','university','know_'+p,'anykids','pref_risk_averse','pref_beq_money','pref_home','totinc']]
    Xs = sm.add_constant(Xs)
    model = sm.OLS(df['delta_'+p],Xs,missing='drop')
    results.append(model.fit())

print(summary_col(results,stars=True))


print(df[['delta_ann','delta_ltc','delta_rmr']].corr())

print(df[['delta_ann','delta_ltc','delta_rmr']].describe())






