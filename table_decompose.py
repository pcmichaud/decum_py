import pandas as pd
import numpy as np



scenarios = ['reference','averse','nobequest','muhealth','house_pref','no_drift','no_risk','obj_surv','obj_home','lowfloor','loading','singles','nomedexp','yaari']
products = ['ann','ltci','rmr']
indp_labels = ['buy_'+p+'_indp' for p in products]

table = pd.DataFrame(index=scenarios,columns=products,dtype='float64')

for scn in scenarios:
    df = pd.read_csv('output/joint_'+scn+'.csv')
    # marginals
    for i in indp_labels:
        df[i] = df[i]>0.0
    takeup = df[indp_labels].mean().values
    table.loc[scn,:] = takeup

table.round(3).to_latex('output/decompose_stats.tex')

print(table.round(3))
