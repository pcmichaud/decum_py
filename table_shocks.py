import pandas as pd
import numpy as np



scenarios = ['reference_noshocks','reference_larger','reference_kappa1','reference_kappa10','reference_kappa250','reference_kappa100','reference_kappa','reference_concave','reference_square','reference_quarter','reference_shocks3','reference_shocks5']

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

table.round(3).to_latex('output/decompose_stats_ez_shocks.tex')

print(table.round(3))
