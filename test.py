from prefs import *
from tools import *
from budget import *
from space import *
from actors import *
from survival import *
import numpy as np
from matplotlib import pyplot as plt

# background of respondents
df_hh = load_hh()
df_rp = load_rp()
df_sp = load_sp()
df_shifters = load_shifters()
df_know = load_know()

# survival and house price appreciation expectations and parameters
df_hp, df_hp_sp, df_survival_bias = load_survival()
df_house_bias = load_house_expectations()

# parameters of the scenarios
df_prices, df_benfs = load_scenarios()

# parameters for costs and house prices (by CMA)
df_nh,df_hc = load_costs()
df_house_prices = load_house_prices()

# pick a guy
i = 159
hh = dict(df_hh.loc[i,:])
rp = dict(df_rp.iloc[i,:])
sp = dict(df_sp.iloc[i,:])
g = df_house_prices.loc[hh['cma'],'g'] * df_house_bias.loc[i,'mu']
sig = df_house_prices.loc[hh['cma'],'sig'] * df_house_bias.loc[i,'zeta']
base = df_house_prices.loc[hh['cma'],'base_value']
home_value = df_hh.loc[i,'home_value']

# create to hold parameters
float_pars, int_pars = create_pars()

# times for problem
int_pars,time_t = times(int_pars)

# set rates
float_pars = set_rates(float_pars,int_pars['n'])

e_space, float_pars, int_pars = set_e_grid(-2.0,2.0,5,int_pars,float_pars)

p_h, f_h, p_r, int_pars = house_prices(g,sig,base,home_value,
                                       e_space,int_pars,float_pars)

d_space, d_h, float_pars, int_pars = set_d_grid(0.0,5,p_h,int_pars,float_pars)

s_ij,s_i,s_j,ij_h,a_i,a_j, int_pars = set_s_grid(hh['married'],int_pars)

y_ij = set_income(hh['married'],rp['totinc'],rp['retinc'],sp['sp_totinc'],
                  sp['sp_retinc'], a_i,a_j, int_pars, float_pars)

w_space, float_pars, int_pars = set_w_grid(0.0,1e3,10,d_h,p_h,y_ij,int_pars,
                                     float_pars)

prices = set_prices(0.0,0.0,0.0,int_pars)
benfs = set_benfs(0.0,0.0,0.0,int_pars)

grid_states, is_adm, adm, to_states, int_pars = set_states(int_pars)

b_its = reimburse_loan(benfs,prices,p_h,int_pars,float_pars)

hc = df_hc.loc[hh['cma'],:].to_numpy()
nh = df_nh.loc[hh['cma'],:].to_numpy()

med_ij = set_medexp(hh['married'],s_i,s_j,hc,nh,int_pars)

b_its = reimburse_loan(benfs,prices,p_h,int_pars,float_pars)

gammas, deltas = parse_surv(df_hp.loc[i,:])

q1 = transition_rates(rp['age'],gammas,deltas,0.0,0.0,0.0,int_pars)

print('life expectancy = ',life_exp(q1,0,int_pars))

gammas, deltas = parse_surv(df_hp_sp.loc[i,:])

q1_sp = transition_rates(sp['sp_age'],gammas,deltas,0.0,0.0,0.0,int_pars)

print('life expectancy = ',life_exp(q1_sp,0,int_pars))

q1_ij = joint_surv_rates(q1,q1_sp,int_pars)

qs_ij = adjust_surv(q1_ij,time_t,int_pars)


xs = [x_fun(0.0,hh['wealth_total'],hh['own'],s_i[0],s_j[0],hh['married'],
        hh['own'],0, p_h[0,0], p_r[0,0], b_its[0,0], med_ij[0],y_ij[0,0],
            int_pars,float_pars,
      prices,benfs) for x in np.linspace(0.0,hh['wealth_total'],1000)]
xs = [x[0] for x in xs]

plt.figure()
plt.plot(np.linspace(0.0,hh['wealth_total'],1000),xs)
plt.show()


