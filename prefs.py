import numpy as np
from numba import njit, int64, float64, void
from numba.experimental import jitclass
from numba.types import Tuple
from budget import *
from space import *

@njit(float64(float64,float64,float64,float64,float64),fastmath=True, cache=True)
def ez_fun(u,ev,beta,varepsilon,gamma):
    present = (1.0 - beta) * (u**(1.0 - varepsilon))
    if ev!=0.0:
        if gamma != 1.0:
            future = (beta) * (ev ** ((1.0 - varepsilon) / (1.0 - gamma)))
        else :
            future = (beta) * (np.exp(ev) ** (1.0 - varepsilon))
    else :
        future = 0.0
    ez = (present + future)**(1.0 / (1.0 - varepsilon))
    return ez

@njit(float64(float64,float64,float64,float64,float64),fastmath=True, cache=True)
def eu_fun(u,ev,beta,varepsilon,gamma):
    present = (u**(1.0-gamma))/(1.0-gamma)
    future = beta * ev
    eu = present + future
    return eu

@njit(float64(float64,float64,float64,float64,float64,float64),fastmath=True,cache=True)
def ces_fun(cons,amen,nu_c,nu_h,rho, eqscale):
    cons_eq = cons
    amen_eq = amen
    ces = (nu_c/eqscale) *(  (cons**(1.0 - rho)) + nu_h * (amen**(1.0 - rho)))**(1.0 / (1.0 - rho))
    return ces

spec_prefs = [
    ('varepsilon',float64),
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
    def __init__(self, varepsilon= 0.261, d_varepsilon=0.028, gamma = 0.403,
                d_gamma = 0.145, rho = 0.811, b_x = 0.045, d_b_x = 0.02, b_k = 2.252e3,
                 nu_c0 = 1.0, nu_c1 = 0.117, nu_c2 = 0.027, nu_h = 0.186, d_nu_h = 0.012, beta = 0.97, live_fast=0, risk_averse=0,
                 beq_money=0, pref_home=0):
        self.varepsilon = varepsilon
        if live_fast==1:
            self.varepsilon -= d_varepsilon
        self.gamma = gamma
        if risk_averse==1:
            self.gamma += d_gamma
        self.rho = rho
        self.b_x = b_x
        if beq_money==1:
            self.b_x += d_b_x
        self.b_k = b_k
        self.nu_c0 = nu_c0
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



