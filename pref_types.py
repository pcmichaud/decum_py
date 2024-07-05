import pandas as pd
import numpy as np


df = pd.read_csv('inputs/pf_rp.csv')



for i in df.index:
    df.loc[i,'pref_type'] = str(df.loc[i,'pref_beq_money']) + str(df.loc[i,'pref_live_fast']) + str(df.loc[i,'pref_risk_averse'])

print(df.mean())
print(df['pref_type'].value_counts())

