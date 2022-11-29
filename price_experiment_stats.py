import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt

q = pd.read_csv('output/price_experiment.csv',index_col=0)
p = q.index.to_list()
plt.figure()
plt.plot(q['ann'],p,label='ANN',color='cornflowerblue')
plt.plot(q['ltci'],p,label='LTCI',color='salmon')
p = np.linspace((0.05-0.03)/0.05,(0.05+0.05)/0.05,5)
plt.plot(q['rmr'],p,label='RMR',color='green')
plt.xlabel('fraction buying')
plt.ylabel('price (cost) relative to fair')
plt.legend()
plt.savefig('output/demand_curves.eps')
plt.show()