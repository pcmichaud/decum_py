
import numpy as np
import pandas as pd
from optim import *
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
# Load values which are compared by run_ref.py
print('annuities:')
df = pd.read_csv('output/delay_ann.csv')
pd.set_option('display.max_rows', 500)
df['buy_ref'] = df['value_delay_0'] > df['value_ref']
alternatives = ['value_ref']
for x in [4,7,10]:
    alternatives.append('value_delay_'+str(x))
df['buy_delay'] = df['value_delay_0'] > df[alternatives].max(axis=1)
print(df[['buy_ref','buy_delay']].mean())

print(df.loc[(df['buy_ref']==True)&(df['buy_delay']==False),['value_ref','value_delay_0','value_delay_10']])
#print(df.loc[:,['value_ref','value_delay_0','value_delay_10']])


print('ltci:')
df = pd.read_csv('output/delay_ltc.csv')
pd.set_option('display.max_rows', 500)
df['buy_ref'] = df['value_delay_0'] > df['value_ref']
alternatives = ['value_ref']
for x in [4,7,10]:
    alternatives.append('value_delay_'+str(x))
df['buy_delay'] = df['value_delay_0'] > df[alternatives].max(axis=1)
print(df[['buy_ref','buy_delay']].mean())


print('rmr')
df = pd.read_csv('output/delay_rmr.csv')
pd.set_option('display.max_rows', 500)
df['buy_ref'] = df['value_delay_0'] > df['value_ref']
alternatives = ['value_ref']
for x in [4,7,10]:
    alternatives.append('value_delay_'+str(x))
df['buy_delay'] = df['value_delay_0'] > df[alternatives].max(axis=1)
print(df[['buy_ref','buy_delay']].mean())
