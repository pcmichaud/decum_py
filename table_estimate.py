import numpy as np
import pandas as pd

# estimates
ssd =  8272.62
sigmas = np.load('output/sigmas_ez.npy')
pars =  np.load('output/estimates_ez.npy')

# standard errors
es = pd.read_csv('output/within_residuals_ez.csv',dtype=np.float64)
es['respid'] = es['respid'].astype('int64')
es.set_index('respid',inplace=True)
gs = pd.read_csv('output/gradients_ez.csv',dtype=np.float64)
gs['respid'] = gs['respid'].astype('int64')
gs.set_index('respid',inplace=True)
gs = gs.drop(labels=['4'],axis=1)
J = len(gs.columns)
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
print(A)
Ainv = np.linalg.inv(A)
cov = Ainv @ B @ Ainv
se = np.sqrt(np.diag(cov))
se = np.delete(se,4)

labels= ['$\\gamma$','$\\varepsilon$',
         '$\\rho$','$b_X$','$\\nu_{c,2}$','$\\nu_{c,3}$',
         '$\\nu_{h}$']

table = pd.DataFrame(index=labels,columns=['point','se'])

table['point'] = np.delete(pars,4)


table.loc['$\\sigma_{\\upsilon,A}(0)$','point'] = sigmas[0,0]
table.loc['$\\sigma_{\\upsilon,A}(1)$','point'] = sigmas[0,1]
table.loc['$\\sigma_{\\upsilon,L}(0)$','point'] = sigmas[1,0]
table.loc['$\\sigma_{\\upsilon,L}(1)$','point'] = sigmas[1,1]
table.loc['$\\sigma_{\\upsilon,R}(0)$','point'] = sigmas[2,0]
table.loc['$\\sigma_{\\upsilon,R}(1)$','point'] = sigmas[2,1]

table['se'] = se

table.loc['within SSE','point'] = ssd

print(table)

table.round(3).to_latex('output/table_estimates_ez.tex',escape=False)




