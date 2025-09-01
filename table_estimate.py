import numpy as np
import pandas as pd
from optim import *
# estimates
ssd =  7885.29
sigmas = np.load('output/sigmas_ez.npy')
sigmas = sigmas[:,0]
pars =  np.load('output/estimates_ez.npy')

isfree = np.ones(pars.shape[0])
isfree[4] = 0
isfree[5] = 0
theta = set_theta(pars,isfree)

# standard errors
es = pd.read_csv('output/within_residuals_ez.csv',dtype=np.float64)
es['respid'] = es['respid'].astype('int64')
es.set_index('respid',inplace=True)
gs = pd.read_csv('output/gradients_ez.csv',dtype=np.float64)
cols = ['respid','scn']
for j in range(9):
    cols.append(str(j))
gs.columns = cols
gs['respid'] = gs['respid'].astype('int64')
gs.set_index(['respid','scn'],inplace=True)
J = len(gs.columns)
print(J)
n = len(es)
A = np.zeros((J,J),dtype=np.float64)
B = np.zeros((J,J),dtype=np.float64)
for i in es.index:
    e_i = es.loc[i,:].to_numpy().reshape((1,12))
    e_i = e_i[np.isnan(e_i)==False]
    e_i = e_i.reshape((1,e_i.shape[0]))
    g_i = gs.loc[i,:].to_numpy()
    ji = g_i.shape[0]
    A = A + g_i.T @ g_i
    B = B + g_i.T @ (e_i.T @ e_i) @ g_i
Ainv = np.linalg.inv(A)
cov = Ainv @ B @ Ainv
se = np.sqrt(np.diag(cov))

print(cov)
# Wald test for EU vs. VNM restriction : gamma = varepsilon
cov_theta = cov[:6,:6]
R = np.array([1,-1,0,0,0,0]).reshape((1,6))
theta = theta.reshape((6,1))
W = (R @ theta).T @ np.linalg.inv(R @ cov_theta @ R.T) @ (R @ theta)
print((R @ theta))
print(W)

labels= ['$\\gamma$','$\\varepsilon$',
         '$\\rho$','$b_X$','$\\nu_{c,3}$',
         '$\\nu_{h}$']

table = pd.DataFrame(index=labels,columns=['point','se'])
ix = [0,1,2,3,6,7]
freepars = pars[ix]
table['point'] = freepars

table.loc['$\\sigma_{\\upsilon,A}$','point'] = sigmas[0]
table.loc['$\\sigma_{\\upsilon,L}$','point'] = sigmas[1]
table.loc['$\\sigma_{\\upsilon,R}$','point'] = sigmas[2]

table['se'] = se

#rescaling

# exponential transformations
table.iloc[0,1]  = table.iloc[0,0]*table.iloc[0,1]
table.iloc[1,1]  = table.iloc[1,0]*table.iloc[1,1]
table.iloc[3,1]  = table.iloc[3,0]*table.iloc[3,1]
table.iloc[4,1]  = table.iloc[4,0]*table.iloc[4,1]
table.iloc[5,1]  = table.iloc[5,0]*table.iloc[5,1]

# logit transform
table.iloc[2,1] = table.iloc[2,0]*(1.0-table.iloc[2,0])*table.iloc[2,1]

table.loc['within SSE','point'] = ssd

print(table)

table.round(3).to_latex('output/table_estimates_ez.tex',escape=False)




