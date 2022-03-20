import numpy as np
import pandas as pd


ssd =  7923.642424856028

sigmas =  [0.032036111577690526, 0.44885211797694485, 0.06445417371634762]

pars =  [ 6.14035846e-01,  1.60346900e-02,  5.83922461e-02,  1.10817068e-01,
  5.15096961e-01,  1.23221463e-03,  7.24697576e-04,  2.61070203e+03,
  6.69292300e-01,  9.40934779e-03,  9.59272905e-02,  1.91240444e-02,
         4.15392802e-03, -8.35684075e-01,  6.00581863e-01]

labels= ['$\\varepsilon$','$\\Delta\\varepsilon$','$\\sigma$','$\\Delta\\sigma$',
         '$\\rho$','$b_x$','$\\Delta b_x$','$b_k$','$\\nu_{c,2}$','$\\nu_{c,3}$',
         '$\\nu_{h,1}$','$\\nu_{h,2}$','$\\Delta \\nu_{h}$','miss r','miss sp']

table = pd.DataFrame(index=labels,columns=['point','se'])

table['point'] = pars
table['se'] = np.nan

table.loc['$\\sigma_{A}$','point'] = sigmas[0]
table.loc['$\\sigma_{L}$','point'] = sigmas[1]
table.loc['$\\sigma_{R}$','point'] = sigmas[2]

table.loc['within SSE','point'] = ssd

print(table)

table.round(3).to_latex('output/table_estimates_ref.tex',escape=False)




