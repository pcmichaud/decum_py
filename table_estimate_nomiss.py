import numpy as np
import pandas as pd

# estimates
ssd =  7980.57
sigmas = np.load('output/sigmas_nomiss.npy')
pars =  np.load('output/estimates_nomiss.npy')

# standard errors
es = pd.read_csv('output/within_residuals_nomiss.csv',dtype=np.float64)
es['respid'] = es['respid'].astype('int64')
es.set_index('respid',inplace=True)
gs = pd.read_csv('output/gradients_nomiss.csv',dtype=np.float64)
gs['respid'] = gs['respid'].astype('int64')
gs.set_index('respid',inplace=True)
gs = gs[[str(x) for x in range(gs.shape[1]-1)]]
print(gs.head())
print(gs.loc[gs.index==10066,:])
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
Ainv = np.linalg.inv(A)
cov = Ainv @ B @ Ainv
se = np.sqrt(np.diag(cov))


se = np.insert(se,7,0.0)
se = np.insert(se,12,0.0)
se = np.insert(se,13,0.0)

labels= ['$\\varepsilon$','$\\varepsilon_{\\Delta}$','$\\gamma$','$\\gamma_{\\Delta}$',
         '$\\rho$','$b$','$b_{\\Delta}$','$b_k$','$\\nu_{c,2}$','$\\nu_{c,3}$',
         '$\\nu_{h}$', '$\\nu_{h,\\Delta}$','$\\psi_{r}$','$\\psi_{s}$']

table = pd.DataFrame(index=labels,columns=['point','se'])

table['point'] = pars

table.loc['$\\varepsilon_{\\Delta}$',:] = - table.loc['$\\varepsilon_{\\Delta}$',:]

table.loc['$\\sigma_{\\upsilon,A}(0)$','point'] = sigmas[0,0]
table.loc['$\\sigma_{\\upsilon,A}(1)$','point'] = sigmas[0,1]
table.loc['$\\sigma_{\\upsilon,L}(0)$','point'] = sigmas[1,0]
table.loc['$\\sigma_{\\upsilon,L}(1)$','point'] = sigmas[1,1]
table.loc['$\\sigma_{\\upsilon,R}(0)$','point'] = sigmas[2,0]
table.loc['$\\sigma_{\\upsilon,R}(1)$','point'] = sigmas[2,1]

table['se'] = se

table.loc['within SSE','point'] = ssd

print(table)

table.round(3).to_latex('output/table_estimates_miss.tex',escape=False)




