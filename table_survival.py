import pandas as pd
import numpy as np



scenarios = ['obj_surv','obj_surv_10','obj_surv_12','obj_surv_30','obj_surv_40','obj_surv_50']

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

table.round(3).to_latex('output/decompose_stats_survival.tex')

print(table.round(3))
