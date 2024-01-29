import pandas as pd
import numpy as np

df = pd.read_csv('output/joint_reference.csv')

# labels
products = ['ann','ltci','rmr']
joint_labels = ['buy_'+p+'_joint' for p in products]
indp_labels = ['buy_'+p+'_indp' for p in products]

# pricing
table = df[['price_ann_fair','price_ltci_fair','price_rmr_fair']].describe().transpose()
table.round(3).to_latex('output/fair_prices.tex')
print(table)

# actual prices in scenarios
prem = df.loc[df['prem_scn_ann_1']!=-999,['prem_scn_ann_'+str(scn) for scn in range(1,5)]].mean().to_list()
ben = df[['ben_scn_ann_'+str(scn) for scn in range(1,5)]].mean().to_list()
print('annuities = ',[p/b for p,b in zip(prem,ben)])

prem = df[['prem_scn_ltci_'+str(scn) for scn in range(1,5)]].mean().to_list()
ben = df[['ben_scn_ltci_'+str(scn) for scn in range(1,5)]].mean().to_list()
print('ltci = ',[p/b for p,b in zip(prem,ben)])

prem = df.loc[df['int_scn_rmr_1']!=-999,['int_scn_rmr_'+str(scn) for scn in range(1,5)]].mean().to_list()
# temp fix until re-estimate, we had added safe rate to rate in reimburse
prem = [p+0.01 for p in prem]
print('rmr = ',prem)




# marginals
print(df[joint_labels].describe().transpose())
print(df[indp_labels].describe().transpose())


# distributions
joint = df[joint_labels].value_counts(normalize=True).to_frame()
indp = df[indp_labels].value_counts(normalize=True).to_frame()
joint.index.names = products
indp.index.names = products
table = joint.merge(indp,left_index=True,right_index=True,how='outer')
table.columns = ['joint','indp']
for c in table.columns:
	table[c] = np.where(table[c].isna(),0.0,table[c])
print(table)

# distributions with all combinations (extensive margin)
for c in joint_labels:
	df[c+'_any'] = df[c]>0
for c in indp_labels:
	df[c+'_any'] = df[c]>0
joint = df[[c+'_any' for c in joint_labels]].value_counts(normalize=True).to_frame()
indp = df[[c+'_any' for c in indp_labels]].value_counts(normalize=True).to_frame()
joint.index.names = products
indp.index.names = products
table = joint.merge(indp,left_index=True,right_index=True,how='outer')
table.columns = ['joint','indp']
for c in table.columns:
	table[c] = np.where(table[c].isna(),0.0,table[c])
print(table)
table.round(3).to_latex('output/joint_distribution_extensive_ez.tex')


