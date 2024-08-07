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

@njit(float64(float64,float64,float64,float64),fastmath=True,cache=True)
def bk_fun(w,b_x,b_k,vareps):
    return b_x/(1-vareps) * ((w+b_k)**(1-vareps) - b_k**(1-vareps))

@njit(float64(float64,float64,float64,float64,float64),fastmath=True,cache=True)
def bo_fun(u,ev,beta,gamma,vareps):
    return (1-beta)*u - beta/gamma * np.log(ev)

@njit(float64(float64,float64,float64,float64, float64),fastmath=True, cache=True)
def ez_fun(u,ev,beta,gamma,vareps):
    present = (1-beta) *(u**(1.0-vareps))
    ezv = ev**((1.0-vareps)/(1.0-gamma))
    future = beta * ezv
    ez = (present + future)**(1.0/(1.0-vareps))
    return ez

@njit(float64(float64,float64,float64,float64, float64),fastmath=True, cache=True)
def bu_fun(w,b_x,b_k,gamma,vareps):
    b_w = max(w + b_k,1.0)
    b_u = (b_x**(vareps/(1.0-vareps))) * b_w
    return b_u

@njit(float64(float64,float64,float64,float64,float64,float64),fastmath=True,cache=True)
def cob_fun(cons,amen,nu_c, vareps, rho, eqscale):
    nu = nu_c/eqscale
    nu = nu**(vareps/(1.0-vareps))
    ces = nu*((cons**rho) * (amen**(1.0-rho)) )
    return ces

spec_prefs = [
    ('gamma',float64),
    ('rho',float64),
    ('b_x',float64),
    ('b_k',float64),
    ('vareps',float64),
    ('nu_c1',float64),
    ('nu_c2',float64),
    ('nu_h',float64),
    ('beta',float64)
]


@jitclass(spec_prefs)
class set_prefs(object):
    def __init__(self, gamma = 0.403,
                d_gamma = 0.145, rho = 0.811, b_x = 0.045, d_b_x = 0.02, b_k = 500.0, vareps = 0.403,
                  nu_c1 = 0.117, nu_c2 = 0.027, nu_h = 0.186, d_nu_h = 0.012, beta = 0.97):
        self.gamma = gamma
        self.rho = rho
        self.b_x = b_x
        self.b_k = b_k
        self.vareps = vareps
        self.nu_c1 = nu_c1
        self.nu_c2 = nu_c2
        self.nu_h = nu_h
        self.beta = beta
        return

@njit(float64[:](int64, int64[:], int64[:],
    set_dims.class_type.instance_type, set_prefs.class_type.instance_type, float64),fastmath=True, cache=True)
def update_nus(married, s_i, s_j, dims, prefs, eqscale):
    nu_ij_c = np.empty(dims.n_s)
    nu_c = np.array([1.0, 1.0, prefs.nu_c2, 0.0])
    if married==1:
        for i in range(dims.n_s):
            nu_ij_c[i] = nu_c[s_i[i]] + nu_c[s_j[i]]
    else :
        for i in range(dims.n_s):
            nu_ij_c[i] = nu_c[s_i[i]]
    return nu_ij_c



