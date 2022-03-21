import numpy as np
import pandas as pd

# estimates
ssd =  7923.642424856028

sigmas =  [0.032036111577690526, 0.44885211797694485, 0.06445417371634762]

pars =  [ 6.14035846e-01,  1.60346900e-02,  5.83922461e-02,  1.10817068e-01,
  5.15096961e-01,  1.23221463e-03,  7.24697576e-04,  2.61070203e+03,
  6.69292300e-01,  9.40934779e-03,  9.59272905e-02,  1.91240444e-02,
         4.15392802e-03, -8.35684075e-01,  6.00581863e-01]

# standard errors 
es = pd.read_csv('output/within_residuals_ref.csv',dtype=np.float64)
es.set_index('respid',inplace=True)
gs = pd.read_csv('output/gradients_ref.csv',dtype=np.float64)
gs.set_index('respid',inplace=True)
J = len(pars) + 3
n = len(es)
A = np.zeros((J,J),dtype=np.float64)
B = np.zeros((J,J),dtype=np.float64)
for i in es.index:
    e_i = es.loc[i,:].to_numpy().reshape((1,12))
    e_i = e_i[np.isnan(e_i)==False]
    e_i = e_i.reshape((1,e_i.shape[0]))
    g_i = gs.loc[(i,),:].to_numpy()
    g_i = g_i[:,1:]
    ji = g_i.shape[0]
    A = A + g_i.T @ g_i
    B = B + g_i.T @ (e_i.T @ e_i) @ g_i
Ainv = np.linalg.inv(A)
cov = Ainv @ B @ Ainv 
se = np.sqrt(np.diag(cov))

labels= ['$\\varepsilon$','$\\Delta\\varepsilon$','$\\sigma$','$\\Delta\\sigma$',
         '$\\rho$','$b_x$','$\\Delta b_x$','$b_k$','$\\nu_{c,2}$','$\\nu_{c,3}$',
         '$\\nu_{h,1}$','$\\nu_{h,2}$','$\\Delta \\nu_{h}$','miss r','miss sp']

table = pd.DataFrame(index=labels,columns=['point','se'])

table['point'] = pars

table.loc['$\\sigma_{A}$','point'] = sigmas[0]
table.loc['$\\sigma_{L}$','point'] = sigmas[1]
table.loc['$\\sigma_{R}$','point'] = sigmas[2]

table['se'] = se 

table.loc['within SSE','point'] = ssd

print(table)

table.round(3).to_latex('output/table_estimates_ref.tex',escape=False)




