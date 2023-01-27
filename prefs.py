import numpy as np
from numba import njit, int64, float64, void
from numba.experimental import jitclass
from numba.types import Tuple
from budget import *
from space import *


@njit(float64(float64,float64,float64,float64),fastmath=True, cache=True)
def eu_fun(u,ev,beta,gamma):
    if gamma!=1.0:
        present = (u**(1.0-gamma))/(1.0-gamma)
    else :
        present = np.log(u)
    future = beta * ev
    eu = present + future
    return eu

@njit(float64(float64,float64,float64,float64),fastmath=True, cache=True)
def bu_fun(w,b_x,b_k,gamma):
    b_w = w + b_k
    if gamma!=1.0:
        b_u = b_x * (b_w**(1.0-gamma))/(1.0-gamma)
    else :
        b_u = b_x * np.log(b_w)
    return b_u

@njit(float64(float64,float64,float64,float64,float64),fastmath=True,cache=True)
def cob_fun(cons,amen,nu_c, rho, eqscale):
    ces = (nu_c/eqscale) *((cons**rho) * (amen**(1.0-rho)) )
    return ces

spec_prefs = [
    ('gamma',float64),
    ('rho',float64),
    ('b_x',float64),
    ('b_k',float64),
    ('nu_c0',float64),
    ('nu_c1',float64),
    ('nu_c2',float64),
    ('nu_h',float64),
    ('beta',float64)
]


@jitclass(spec_prefs)
class set_prefs(object):
    def __init__(self, gamma = 0.403,
                d_gamma = 0.145, rho = 0.811, b_x = 0.045, d_b_x = 0.02, b_k = 500.0,
                  nu_c1 = 0.117, nu_c2 = 0.027, nu_h = 0.186, d_nu_h = 0.012, beta = 0.97, risk_averse=0,
                 beq_money=0, pref_home=0):
        self.gamma = gamma
        if risk_averse==1:
            self.gamma += d_gamma
        self.rho = rho
        self.b_x = b_x
        if beq_money==1:
            self.b_x += d_b_x
        self.b_k = b_k
        self.nu_c0 = 1.0
        self.nu_c1 = nu_c1
        self.nu_c2 = nu_c2
        self.nu_h = nu_h
        if pref_home==1:
            self.nu_h += d_nu_h
        self.beta = beta
        return

@njit(float64[:](int64, int64[:], int64[:],
    set_dims.class_type.instance_type, set_prefs.class_type.instance_type, float64),fastmath=True, cache=True)
def update_nus(married, s_i, s_j, dims, prefs, eqscale):
    nu_ij_c = np.empty(dims.n_s)
    nu_c = np.array([prefs.nu_c0, prefs.nu_c1, prefs.nu_c2, 0.0])
    if married==1:
        for i in range(dims.n_s):
            nu_ij_c[i] = nu_c[s_i[i]] + nu_c[s_j[i]]
    else :
        for i in range(dims.n_s):
            nu_ij_c[i] = nu_c[s_i[i]]
    return nu_ij_c



