import numpy as np
from numba import njit, float64, int64
from numba.types import Tuple
from numba.experimental import jitclass
import pandas as pd
from prefs import *
from space import *

spec_rates = [
    ('rate',float64),
    ('r_r',float64),
    ('r_d',float64),
    ('r_h',float64),
    ('xi_d',float64),
    ('phi',float64),
    ('phi_d',float64),
    ('x_min',float64),
    ('tau_s0',float64),
    ('tau_s1',float64),
    ('tau_b0',float64),
    ('tau_b1',float64),
    ('omega_d',float64),
    ('omega_rm',float64),
    ('omega_r',float64),
    ('omega_h0',float64),
    ('omega_h1',float64),
    ('eqscale',float64)
]





@jitclass(spec_rates)
class set_rates(object):
    def __init__(self, rate=0.01, r_r=0.0949, r_d=0.02, r_h=0.01,
              xi_d=0.9622,phi_d = 0.1,x_min = 18.2, tau_s0 = 1.5,tau_s1 = 0.05,
              tau_b0 = 0.5,tau_b1 = 0.01, omega_d = 0.65, omega_rm = 0.55,
              omega_r = 0.329, omega_h0 = 0.65,omega_h1 = 0.8,
              phi = 0.035, eqscale = 0.55):
        self.rate = rate
        self.r_r = r_r
        self.r_d = self.rate + r_d
        self.r_h = r_h  + self.r_d
        self.xi_d = xi_d
        self.phi = phi
        self.phi_d = phi_d
        self.x_min = x_min
        self.tau_s0 = tau_s0
        self.tau_s1 = tau_s1
        self.tau_b0 = tau_b0
        self.tau_b1 = tau_b1
        self.omega_d = omega_d
        self.omega_rm = omega_rm
        self.omega_r = omega_r
        self.omega_h0 = omega_h0
        self.omega_h1 = omega_h1
        self.eqscale = eqscale
        return


spec_prices = [
    ('ann',float64),
    ('ltc',float64),
    ('rmr',float64)
]

@jitclass(spec_prices)
class set_prices(object):
    def __init__(self, ann, ltc, rmr):
        self.ann = ann
        self.ltc = ltc
        self.rmr = rmr
        return

spec_benfs = [
    ('ann',float64),
    ('ltc',float64),
    ('rmr',float64)
]

@jitclass(spec_benfs)
class set_benfs(object):
    def __init__(self,ann, ltc, rmr):
        self.ann = ann
        self.ltc = ltc
        self.rmr = rmr
        return

@njit(float64(float64,float64,int64,float64,float64,float64,float64),
    fastmath=True, cache=True)
def beq_fun(d, w, i_hh, p_h, b_its, tau_s0, tau_s1):
    beq = w
    if i_hh==1:
        mc_s = tau_s0 + tau_s1 * p_h
        p = p_h - d - mc_s - b_its
        beq += p
    beq = max(beq,0.01)
    return beq

@njit(Tuple((float64,float64))(float64,float64,int64,int64,
    int64,int64,int64,int64,float64,float64,float64,float64,float64,
     set_dims.class_type.instance_type, set_rates.class_type.instance_type,
     set_prices.class_type.instance_type, set_benfs.class_type.instance_type),
      fastmath=True, cache=True)
def x_fun(d0, w0, h0, s_i, s_j, marr, h1, tt, p_h, p_r, b_its, med, y,
          dims, rates, prices, benfs):
    d1 = h1 * (rates.xi_d * h0 * d0 + (1.0 - h0) * rates.omega_d * p_h)

    c_h = (1.0 - h1) * p_r + h1 * p_h - d1
    mc_s = rates.tau_s0 + rates.tau_s1 * p_h
    mc_b = rates.tau_b0 + rates.tau_b1 * p_h
    mc = h0 * (1.0 - h1) * mc_s + (1.0 - h0) * h1 * mc_b
    w_h = h0 * (p_h - d0 * np.exp(rates.r_d))
    if tt==0:
        z = - prices.ann - prices.ltc  \
            + h0 * h1  * benfs.rmr
    else :
        z = -h0 * (1.0 - h1) * b_its
        if s_i <= 2:
            z += benfs.ann
        if s_i == 2:
            z += benfs.ltc
        if s_i < 2:
            z -= prices.ltc
    x = w0 + w_h + y + z - med
    x_f = 0.0
    if s_i<=2:
        x_f += rates.x_min
    if marr == 1:
        if s_j<=2:
            x_f += rates.x_min * rates.eqscale
    tr = max(x_f - x + (1.0-h1) * p_r, 0.0)
    x += tr - c_h - mc
    return x, tr

@njit(float64[:,:](set_benfs.class_type.instance_type,set_prices.class_type.instance_type,float64[:,:],
        set_dims.class_type.instance_type, set_rates.class_type.instance_type),fastmath=True, cache=True)
def reimburse_loan(benfs,prices,p_h,dims,rates):
    b_its = np.empty((dims.n_e,dims.T))
    pi_r = prices.rmr
    for i in range(dims.T):
        for j in range(dims.n_e):
            b_its[j,i] = min(benfs.rmr * np.exp(pi_r*float(i)),p_h[j,i])
    return b_its

def load_house_prices(file_d='house_prices_real.csv',file_b='home_values.csv'):
    df = pd.read_csv('inputs/'+file_d)
    df.columns = ['cma_name','g','sig','pval']
    df = df.drop(labels='pval',axis=1)
    df['cma'] = np.arange(1,12)
    df.set_index('cma',inplace=True)
    df_b = pd.read_csv('inputs/'+file_b,header=None)
    df_b.columns = ['cma','base_value']
    df_b.set_index('cma',inplace=True)
    df = df.merge(df_b,left_index=True,right_index=True,how='left')
    df['base_value'] *= 1e-3
    return df

@njit(Tuple((float64[:,:],float64[:],float64[:,:]))(float64, float64, float64,
    float64, set_rates.class_type.instance_type, set_dims.class_type.instance_type), fastmath=True, cache=True)
def house_prices(g,sig,base_h,home_value,rates,dims):
    mu = np.exp(g + 0.5*sig**2)
    if home_value>0.0:
        p_h_0 = home_value
    else :
        p_h_0 = base_h
    omega = np.exp(2.0*g + 2.0*(sig**2)) - np.exp(2.0*g + sig**2)
    p_h = np.empty((dims.n_e,dims.T),dtype=np.float64)
    for i in range(dims.T):
        if i==0:
            p_h[:, 0] = p_h_0
        else :
            for j in range(dims.n_e):
                e_p = p_h_0 * (mu**i)
                v_p = (p_h_0**2) * ((omega + mu**2)**i - (mu ** (2*i)))
                p_h[j, i] = max(e_p + np.sqrt(v_p) * dims.e_space[j],50.0)
    f_h = np.empty(dims.n_e,dtype=np.float64)
    for j in range(dims.n_e):
        f_h[j] = (1.0/np.sqrt(2.0*3.14159)) * np.exp(-0.5*(dims.e_space[j]**2))
    f_h = f_h/np.sum(f_h)
    p_r = rates.phi * p_h
    return p_h, f_h, p_r

@njit(float64[:,:](int64, float64, float64, float64, float64,
    set_dims.class_type.instance_type),fastmath=True, cache=True)
def set_income(married,totinc,retinc,sp_totinc,sp_retinc,dims):
    y = np.empty(dims.T)
    y[:] = retinc
    y[0] = totinc
    sp_y = np.empty(dims.T)
    if married ==1:
        sp_y[:] = sp_retinc
        sp_y[0] = sp_totinc
    else :
        sp_y[:] = 0.0
    y_ij = np.empty((dims.n_s, dims.T))
    for i in range(dims.n_s):
        for j in range(dims.T):
            y_ij[i,j] = dims.a_i[i] * y[j]
            if married==1:
                y_ij[i,j] += dims.a_j[i] * sp_y[j]
    return y_ij

@njit(float64[:](int64,float64[:],float64[:],
                 set_dims.class_type.instance_type),
      fastmath=True, cache=True)
def set_medexp(married, hc, nh, dims):
    med_ij = np.empty(dims.n_s,dtype=np.float64)
    for i in range(dims.n_s):
        if married==0:
            if dims.s_i[i]==0:
                med_ij[i] = hc[0]
            elif dims.s_i[i]==1:
                med_ij[i] = hc[1]
            elif dims.s_i[i]==2:
                med_ij[i] = hc[2] + nh[1]
            else :
                med_ij[i] = 0.0
        else :
            n_hc = 0
            if dims.s_i[i]==1:
                n_hc +=1
            if dims.s_j[i]==1:
                n_hc +=1
            n_nh = 0
            if dims.s_i[i]==2:
                n_nh +=1
            if dims.s_j[i]==2:
                n_nh +=1
            med_ij[i] = hc[n_hc] + nh[n_nh]
            if i==dims.n_s-1:
                med_ij[i] = 0.0
    return med_ij

def load_costs(file_nh='ncare_costs.csv',file_hc='hcare_costs.csv'):
    df_nh = pd.read_csv('inputs/'+file_nh,header=None,
                        dtype='float64')
    df_hc = pd.read_csv('inputs/'+file_hc,header=None,
                        dtype='float64')
    df_nh.columns = [0,1,2]
    df_hc.columns = [0,1,2]
    df_nh['cma'] = np.arange(1,12)
    df_nh.set_index('cma',inplace=True)
    df_hc['cma'] = np.arange(1,12)
    df_hc.set_index('cma',inplace=True)
    for c in df_hc.columns:
        df_hc[c] *= 1e-3
    for c in df_nh.columns:
        df_nh[c] *= 1e-3
    return df_nh, df_hc
