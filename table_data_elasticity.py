#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from linearmodels import PanelOLS
import statsmodels.api as sm



# # Load Data and Express Long-Form

# In[89]:

prices = pd.read_csv('inputs/prices.csv')
prices.set_index('respid',inplace=True)
prices.columns = [x for x in range(1,13)]
benfs = pd.read_csv('inputs/benefits.csv')
benfs.set_index('respid',inplace=True)
benfs.columns = [x for x in range(1,13)]
prob = pd.read_csv('inputs/prob.csv')
prob.set_index('respid',inplace=True)
prob.columns = [x for x in range(1,13)]


# In[90]:


# set in long form
prices = prices.stack()
benfs = benfs.stack()
prob = prob.stack()


# In[91]:


data = pd.concat([prices,benfs,prob],axis=1)
data.columns = ['price','benfs','prob']
data.head()


# In[92]:


for c in data.columns:
	data.loc[data[c]==-999,c] = np.nan
data.dropna(axis=0,inplace=True)


# In[93]:


data.head()


# In[ ]:





# In[97]:


products = ['ann','ltc','rmr']
pairs = [(1,5),(5,9),(9,13)]
table = pd.DataFrame(index=['prob buy','all zeros','price e','ben e'],columns=products)
for i,p in enumerate(pairs):
	df = data.loc[data.index.get_level_values(1).isin(np.arange(p[0],p[1])),:]
	df = df.loc[:,['prob','benfs','price']]
	#df.loc[df.prob==0,'prob'] = np.nan
	df = df.dropna(axis=0)
	#df = df[df.benfs!=0]
	print(df.describe())
	y = df.loc[:,'prob']
	X = df.loc[:,['benfs','price']]
	x_mean = X.mean(axis=0)
	y_mean = y.mean()
	X = sm.add_constant(X)
	table.loc['prob buy',products[i]] = df['prob'].mean()
	mod = PanelOLS(y,X,entity_effects=True)
	results = mod.fit()
	table.loc['ben e',products[i]] = results.params[1]*x_mean[0]/y_mean
	table.loc['price e',products[i]] = results.params[2]*x_mean[1]/y_mean
	df['prob_zero'] = df.loc[:,'prob']==0
	table.loc['all zeros',products[i]] = (df.groupby('respid').sum()['prob_zero']==4).mean()


# In[98]:


table


# In[102]:


know = pd.read_csv('inputs/know.csv')
know.set_index('respid',inplace=True)


# In[104]:


means = know.mean()
table.loc['know',:] = means.to_list()


# In[106]:


for c in table.columns:
	table[c] = table[c].astype('float64')


# In[107]:


table.round(3).to_latex('output/choice_probabilities.tex')

print(table)

# In[ ]:




