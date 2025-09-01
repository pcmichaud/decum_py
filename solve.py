import numpy as np
from tools import *
from actors import *
from survival import *
from prefs import *
from numba import njit, float64, int64
from numba.types import Tuple
import os, time
from functools import partial
from scipy.optimize import golden
from math import floor

def setup_problem(hh, rp, sp, g, sig, phi, tau_b,  base_value, hc, nh, hp, hp_sp, surv_bias,
                  sp_surv_bias,miss_par=0.0,sp_miss_par=0.0, h = 0.0):
    # create rates
    rates = set_rates(phi = phi, tau_b1 = tau_b)
    # create dimensions
    dims = set_dims(hh['married'], rates.omega_d)
    # house price dynamics and matrices
    p_h, f_h, p_r = house_prices(g, sig, base_value, hh['home_value'], rates,
                                 dims)
    # finish up state space
    dims.set_dspace(p_h)
    # rates of return (gauss-hermite points and weights)
    dims.r_space, dims.r_wgt = herm_rates(dims.n_r,rates.eq_prem,rates.sd_prem)
    # income
    if hh['married'] == 1:
        y_ij = set_income(hh['married'], rp['totinc'], rp['retinc'],
                          sp['sp_totinc'], sp['sp_retinc'], dims)
    else :
        y_ij = set_income(hh['married'], rp['totinc'], rp['retinc'],
                          0.0, 0.0, dims)
    # wealth grid
    dims.set_wspace(y_ij, p_h, rates)
    # health expenditures
    med_ij = set_medexp(hh['married'], hc, nh, dims)
    # health transitions
    gammas, deltas = parse_surv(hp)
    q1 = transition_rates(rp['age'], gammas, deltas, surv_bias['xi'],
                          surv_bias['miss_psurv85'], miss_par, dims.T,h=h)
    if hh['married'] == 1:
        gammas, deltas = parse_surv(hp_sp)
        q1_sp = transition_rates(sp['sp_age'], gammas, deltas,
                                 sp_surv_bias['xi_sp'],
                                 sp_surv_bias['sp_miss_psurv85'], sp_miss_par, dims.T,h=h)
        q1_ij = joint_surv_rates(q1, q1_sp, dims.n_s, dims.T)
    else :
        q1_ij = q1[:, :, :]
    return p_h, f_h, p_r, y_ij, med_ij, q1_ij, dims, rates

def get_rules(hh, rp, sp, base_value, prices, benfs,
              p_h, f_h, p_r, y_ij, med_ij, qs_ij, b_its, nu_ij_c,
              rates, dims, prefs, v_ref, delay_yrs = 0):
    v_t = np.zeros((dims.n_states, dims.T),dtype='float64')
    c_t = np.zeros((dims.n_states, 2, dims.T),dtype='float64')
    condv_t = np.zeros((dims.n_states, 2, dims.T),dtype='float64')
    c_last = np.zeros((dims.n_states, 2),dtype='float64')
    cc_last = np.zeros((dims.n_states, 2),dtype='float64')
    nextv = v_t[:,dims.T-1].reshape((dims.n_d,dims.n_w,dims.n_s,
                                          dims.n_e,2),order='F')
    # solve final year for admissible states
    v_last, c_last, condv_last = core_fun(dims.t_last, p_h, p_r, b_its,
                f_h, nu_ij_c, med_ij, y_ij, qs_ij, hh['married'],
                base_value, dims, rates, prefs, prices, benfs, nextv, nextv,  cc_last, 1, delay_yrs)

    # map those to all states
    v_t[dims.to_states[:],dims.t_last] = v_last[:]
    c_t[dims.to_states[:],0,dims.t_last] = c_last[:,0]
    c_t[dims.to_states[:],1,dims.t_last] = c_last[:,1]
    condv_t[dims.to_states[:],0,dims.t_last] = condv_last[:,0]
    condv_t[dims.to_states[:],1,dims.t_last] = condv_last[:,1]
    # solve remaining years
    for t in reversed(range(dims.t_last)):
        isolve = 0
        if t in dims.time_t:
            isolve = 1
        nextv = v_t[:,t+1].reshape((dims.n_d,dims.n_w,dims.n_s,
                                          dims.n_e,2),order='F')
        nextv_ref = v_ref[:,t+1].reshape((dims.n_d,dims.n_w,dims.n_s,dims.n_e,2),order='F')

        cc_last = np.copy(c_last)
        v_last, c_last, condv_last = core_fun(t, p_h, p_r,b_its,
                f_h, nu_ij_c, med_ij, y_ij, qs_ij, hh['married'],
                base_value, dims, rates, prefs, prices, benfs, nextv, nextv_ref,  cc_last, isolve, delay_yrs)
        v_t[dims.to_states[:],t] = v_last[:]
        c_t[dims.to_states[:],0,t] = c_last[:,0]
        c_t[dims.to_states[:],1,t] = c_last[:,1]
        condv_t[dims.to_states[:],0,t] = condv_last[:,0]
        condv_t[dims.to_states[:],1,t] = condv_last[:,1]
    return c_t, condv_t, v_t

def get_value(hh, rp, sp, base_value, prices, benfs,
              p_h, f_h, p_r, y_ij, med_ij, qs_ij, b_its, nu_ij_c,
              rates, dims, prefs, v_ref, delay_yrs = 0):
    v_t = np.zeros((dims.n_states, dims.T),dtype='float64')
    c_t = np.zeros((dims.n_states, 2, dims.T),dtype='float64')
    condv_t = np.zeros((dims.n_states, 2, dims.T),dtype='float64')
    c_last = np.zeros((dims.n_states, 2),dtype='float64')
    cc_last = np.zeros((dims.n_states, 2),dtype='float64')
    nextv = v_t[:,dims.T-1].reshape((dims.n_d,dims.n_w,dims.n_s,
                                          dims.n_e,2),order='F')
    # solve final year for admissible states
    v_last, c_last, condv_last = core_fun(dims.t_last, p_h, p_r, b_its,
                f_h, nu_ij_c, med_ij, y_ij, qs_ij, hh['married'],
                base_value, dims, rates, prefs, prices, benfs, nextv, nextv,  cc_last, 1, delay_yrs)
    #print('done with last year',v_last[:50])
    # map those to all states
    v_t[dims.to_states[:],dims.t_last] = v_last[:]
    c_t[dims.to_states[:],0,dims.t_last] = c_last[:,0]
    c_t[dims.to_states[:],1,dims.t_last] = c_last[:,1]
    condv_t[dims.to_states[:],0,dims.t_last] = condv_last[:,0]
    condv_t[dims.to_states[:],1,dims.t_last] = condv_last[:,1]
    # solve remaining years
    for t in reversed(range(dims.t_last)):
        isolve = 0
        if t in dims.time_t:
            isolve = 1

        nextv = v_t[:,t+1].reshape((dims.n_d,dims.n_w,dims.n_s,
                                          dims.n_e,2),order='F')
        nextv_ref = v_ref[:,t+1].reshape((dims.n_d,dims.n_w,dims.n_s,dims.n_e,2),order='F')
        cc_last = np.copy(c_last)
        v_last, c_last, condv_last = core_fun(t, p_h, p_r,b_its,
                f_h, nu_ij_c, med_ij, y_ij, qs_ij, hh['married'],
                base_value, dims, rates, prefs, prices, benfs, nextv, nextv_ref, cc_last, isolve, delay_yrs)
        v_t[dims.to_states[:],t] = v_last[:]
        # check if delay is optimal when delay positive (only v_t is used, not others)
        #if delay_yrs>0 and t==delay_yrs:
        #    for s in range(dims.n_states):
        #        if v_t[s,t]<v_ref[s,t]:
        #            v_t[s,t] = v_ref[s,t]
        c_t[dims.to_states[:],0,t] = c_last[:,0]
        c_t[dims.to_states[:],1,t] = c_last[:,1]
        condv_t[dims.to_states[:],0,t] = condv_last[:,0]
        condv_t[dims.to_states[:],1,t] = condv_last[:,1]
    # find current state
    h_init = hh['own']
    e_init = int(floor(dims.n_e/2))
    s_init= rp['hlth']-1
    if hh['married']==1:
        s_init = s_init * 4 + sp['sp_hlth'] - 1
    if h_init==1:
        d_low, d_up, du = scale(hh['mort_balance'], dims.d_h[:,e_init,0])
    else :
        d_low = 0
        d_up = 0
        du = 0.0
    ww_space = dims.w_space[d_up,:,s_init,e_init,h_init,0]
    w_low, w_up, wu = scale(hh['wealth_total'],ww_space)
    v = v_t[:, 0].reshape((dims.n_d, dims.n_w, dims.n_s,
                                        dims.n_e, 2),order='F')
    vsub = np.empty((2,2),dtype=np.float64)
    vsub[0,0] = v[d_low,w_low,s_init,e_init,h_init]
    vsub[0,1] = v[d_low,w_up,s_init,e_init,h_init]
    vsub[1,0] = v[d_up,w_low,s_init,e_init,h_init]
    vsub[1,1] = v[d_up,w_up,s_init,e_init,h_init]
    value = interp2d(du, wu, vsub)
    return value, v_t

def get_sim_path(seed, cons_rules, cond_values, hh, rp, sp, base_value, prices, benfs,
              p_h, f_h, p_r, y_ij, med_ij, qs_ij, b_its, nu_ij_c,
              rates, dims, prefs):
    # for solution, set array for path
    cons_path = np.empty(dims.T,dtype=np.float64)
    cons_path[:] = np.nan
    own_path = np.empty(dims.T,dtype=np.float64)
    own_path[:] = np.nan
    wlth_path = np.empty(dims.T, dtype=np.float64)
    wlth_path[:] = np.nan
    home_path = np.empty(dims.T, dtype=np.float64)
    home_path[:] = np.nan
    hlth_path = np.empty(dims.T,dtype=np.float64)
    hlth_path[:] = np.nan

    # find current state
    i_h = hh['own']
    i_e = int(floor(dims.n_e/2))
    i_s = rp['hlth']-1
    d_t = hh['mort_balance']
    w_t = hh['wealth_total']
    if hh['married']==1:
        i_s = i_s * 4 + sp['sp_hlth'] - 1
    vopt = np.zeros(2,dtype=np.float64)
    copt = np.zeros(2,dtype=np.float64)
    sub = np.empty((2,2),dtype=np.float64)
    np.random.seed()
    for t in range(dims.T):
        # figure out where in continuous space
        if i_h==1:
            d_low, d_up, du = scale(d_t, dims.d_h[:,i_e,t])
        else :
            d_low = 0
            d_up = 0
            du = 0.0
        ww_space = dims.w_space[d_low,:,i_s,i_e,i_h,t]
        wlth_path[t] = max(w_t,ww_space[0])
        home_path[t] = i_h*(p_h[i_e,t] - d_t)
        hlth_path[t] = i_s
        w_low, w_up, wu = scale(w_t,ww_space)
        # get decisions by interpolation over continuous state
        for i_hh in range(2):
            v = cond_values[:,i_hh, t].reshape((dims.n_d, dims.n_w, dims.n_s,
                                            dims.n_e, 2),order='F')
            sub[0,0] = v[d_low,w_low,i_s,i_e,i_h]
            sub[0,1] = v[d_low,w_up,i_s,i_e,i_h]
            sub[1,0] = v[d_up,w_low,i_s,i_e,i_h]
            sub[1,1] = v[d_up,w_up,i_s,i_e,i_h]
            vopt[i_hh] = interp2d(du, wu, sub)
        if vopt[1] > vopt[0]:
            own_path[t] = 1
        else :
            own_path[t] = 0
        c = cons_rules[:,int(own_path[t]), t].reshape((dims.n_d, dims.n_w, dims.n_s,
                                            dims.n_e, 2),order='F')
        sub[0,0] = c[d_low,w_low,i_s,i_e,i_h]
        sub[0,1] = c[d_low,w_up,i_s,i_e,i_h]
        sub[1,0] = c[d_up,w_low,i_s,i_e,i_h]
        sub[1,1] = c[d_up,w_up,i_s,i_e,i_h]
        cons_path[t] = interp2d(du, wu, sub)
        # update state to next year
        i_hh = int(own_path[t])
        # update wealth
        cash = x_fun(d_t,w_t,i_h,dims.s_i[i_s],dims.s_j[i_s],hh['married'],i_hh,
                t, p_h[i_e,t],p_r[i_e,t], b_its[i_e,t], med_ij[i_s], y_ij[i_s,t],
                dims,rates, prices, benfs,0)
        w_p = cash[0] - cons_path[t]
        if w_p >= 0.0:
            w_p *= np.exp(rates.rate)
        else :
            r_b = i_h*rates.r_h + (1.0-i_h)*rates.r_r
            w_p *= np.exp(r_b)
        # update mortgage
        d_p = i_hh*(rates.xi_d*i_h*d_t
                    + (1.0 - i_h)*rates.omega_d*p_h[i_e,t])
        # change in house price shock
        i_ee = np.random.choice(np.arange(dims.n_e),p=f_h)
        # determine change in health state
        q_ss = qs_ij[i_s,:,t]
        i_ss = np.random.choice(dims.s_ij,p=q_ss)
        if i_ss==(dims.n_s-1):
            # check if household stil exist
            break
        else :
            # make switch, given survival
            i_h = i_hh
            i_e = i_ee
            i_s = i_ss
            w_t = w_p
            d_t = d_p
    return cons_path, own_path, wlth_path, home_path, hlth_path


@njit(float64(float64,float64,int64,int64,float64[:,:],
              float64[:,:],float64[:],float64[:],
              float64,set_prefs.class_type.instance_type,
              set_dims.class_type.instance_type,
              set_rates.class_type.instance_type, int64),fastmath=True, cache=True)
def v_t_fun(cons, x, z, i_hh, p_h, b_its, f_h, nu_ij_c,
            base_value, prefs, dims, rates, delay_yrs = 0):
    t = dims.t_last
    i_d,i_w,i_s,i_e, i_h = dims.adm[z,:]
    d0 = dims.d_h[i_d,i_e,t]
    w_t = x - cons
    if w_t >= 0.0:
        w_t *= np.exp(rates.rate)
    else :
        r_b = i_h*rates.r_h + (1.0-i_h)*rates.r_r
        w_t *= np.exp(r_b)
    d_t = i_hh*(rates.xi_d*i_h*d0
                + (1.0 - i_h)*rates.omega_d*p_h[i_e,t])
    beq = 0.0
    if prefs.b_x>0.0:
        for i_ee in range(dims.n_e):
            for i_rr in range(dims.n_r):
                ww_t = w_t
                if w_t>0.0:
                    ww_t *= np.exp(rates.share_r*dims.r_space[i_rr])
                b_e = beq_fun(d_t,ww_t,i_hh,p_h[i_ee,t],
                          b_its[i_ee,t],rates.tau_s0,rates.tau_s1)
            beq += f_h[i_ee]*dims.r_wgt[i_rr]*(bu_fun(b_e,prefs.b_x,prefs.b_k,prefs.gamma,prefs.vareps))**(1.0-prefs.gamma)
    amen = (rates.phi + prefs.nu_h * i_h) * p_h[2, 0]
    eqscale = 1.0
    if dims.n_s==16 and dims.a_j[i_s]==1:
        eqscale += rates.eqscale
    #if cons<=0.0:
        #print('got a zero cons in last period',cons, x)
    u = cob_fun(cons, amen, nu_ij_c[i_s], prefs.vareps, prefs.rho, eqscale)
    vopt = ez_fun(u,beq,prefs.beta,prefs.gamma, prefs.vareps)
    return vopt

@njit(float64(float64,float64,int64, int64, int64,float64[:,:],
              float64[:,:],float64[:],float64[:],float64[:,:,:],
              float64, set_prefs.class_type.instance_type,
              set_dims.class_type.instance_type,
              set_rates.class_type.instance_type, float64[:,:,:,:,:],int64),
      fastmath=True, cache=True)
def v_fun(cons, x, z, t, i_hh, p_h, b_its, f_h, nu_ij_c,
            qs_ij, base_value, prefs, dims, rates, nextv, delay_yrs = 0):
    i_d,i_w,i_s,i_e, i_h = dims.adm[z,:]
    d0 = dims.d_h[i_d,i_e,t]
    w_t = x - cons
    if w_t >= 0.0:
        w_t *= np.exp(rates.rate)
    else :
        r_b = i_h*rates.r_h + (1.0-i_h)*rates.r_r
        w_t *= np.exp(r_b)
    d_t = i_hh*(rates.xi_d*i_h*d0
                + (1.0 - i_h)*rates.omega_d*p_h[i_e,t])
    q_ss = qs_ij[i_s,:,t]
    ev = 0.0
    for i_ee in range(dims.n_e):
        d_grid = dims.d_h[:,i_ee,t+1]
        d_low, d_up, du = scale(d_t, dims.d_h[:,i_ee,t+1])
        for i_ss in range(dims.n_s):
            for i_rr in range(dims.n_r):
                ww_t = w_t
                if ww_t>0.0:
                    ww_t *= np.exp(rates.share_r*dims.r_space[i_rr])
                beq = beq_fun(d_t,ww_t,i_hh,p_h[i_ee,t+1],b_its[i_ee,t+1],
                              rates.tau_s0, rates.tau_s1)
                v = nextv[:,:,i_ss,i_ee,i_hh]
                if i_ss < (dims.n_s-1):
                    w_grid_low = dims.w_space[d_low,:,i_ss,i_ee,i_hh,t+1]
                    if ww_t<=w_grid_low[0]:
                        pv_low = v[d_low,0]
                    elif ww_t>=w_grid_low[dims.n_w-1]:
                        pv_low = v[d_low,dims.n_w-1]
                    else :
                        pv_low = cubic_interp1d(w_grid_low, v[d_low,:], ww_t)
                        if pv_low<0.0:
                            w_low, w_up, wu = scale(ww_t, w_grid_low)
                            pv_low = wu*v[d_low,w_up] + (1-wu)*v[d_low,w_low]
                    if d_t!=0.0:
                        w_grid_up = dims.w_space[d_up,:,i_ss,i_ee,i_hh,t+1]
                        if ww_t<=w_grid_up[0]:
                            pv_up = v[d_up,0]
                        elif ww_t>=w_grid_up[dims.n_w-1]:
                            pv_up = v[d_up,dims.n_w-1]
                        else :
                            pv_up = cubic_interp1d(w_grid_up, v[d_up,:], ww_t)
                        if pv_up<0.0:
                            w_low, w_up, wu = scale(ww_t, w_grid_up)
                            pv_up = wu*v[d_up,w_up] + (1-wu)*v[d_up,w_low]
                    else :
                        pv_up = pv_low
                    pv = du*pv_up + (1-du)*pv_low
                    ev += f_h[i_ee] * q_ss[i_ss] * dims.r_wgt[i_rr]* (pv**(1.0-prefs.gamma))
                else :
                    if prefs.b_x > 0.0:
                        bu = bu_fun(beq,prefs.b_x,prefs.b_k,prefs.gamma,prefs.vareps)
                        ev += f_h[i_ee]*q_ss[i_ss] * dims.r_wgt[i_rr]* (bu**(1.0-prefs.gamma))
    amen = (rates.phi + prefs.nu_h * i_h) * p_h[2, 0]
    eqscale = 1.0
    if dims.n_s==16 and dims.a_j[i_s]==1:
        eqscale += rates.eqscale
    u = cob_fun(cons, amen, nu_ij_c[i_s],prefs.vareps,  prefs.rho,eqscale)
    vopt = ez_fun(u,ev,prefs.beta,prefs.gamma, prefs.vareps)
    return vopt

@njit(float64(float64,float64,int64,int64,float64[:,:],
              float64[:,:],float64[:],float64[:],
              float64,set_prefs.class_type.instance_type,
              set_dims.class_type.instance_type,
              set_rates.class_type.instance_type),fastmath=True, cache=True)
def v_t_bo_fun(cons, x, z, i_hh, p_h, b_its, f_h, nu_ij_c,
            base_value, prefs, dims, rates):
    t = dims.t_last
    i_d,i_w,i_s,i_e, i_h = dims.adm[z,:]
    d0 = dims.d_h[i_d,i_e,t]
    w_t = x - cons
    if w_t >= 0.0:
        w_t *= np.exp(rates.rate)
    else :
        r_b = i_h*rates.r_h + (1.0-i_h)*rates.r_r
        w_t *= np.exp(r_b)
    d_t = i_hh*(rates.xi_d*i_h*d0
                + (1.0 - i_h)*rates.omega_d*p_h[i_e,t])
    beq = 0.0
    if prefs.b_x>0.0:
        for i_ee in range(dims.n_e):
            b_e = beq_fun(d_t,w_t,i_hh,p_h[i_ee,t],
                          b_its[i_ee,t],rates.tau_s0,rates.tau_s1)
            beq += f_h[i_ee]*np.exp(-prefs.gamma*(1-prefs.beta)*bk_fun(b_e,prefs.b_x,prefs.b_k,prefs.vareps))
    amen = (rates.phi + prefs.nu_h * i_h) * p_h[2, 0]
    eqscale = 1.0
    if dims.n_s==16 and dims.a_j[i_s]==1:
        eqscale += rates.eqscale
    #if cons<=0.0:
        #print('got a zero cons in last period',cons, x)
    u = cob_fun(cons, amen, nu_ij_c[i_s], prefs.vareps, prefs.rho, eqscale)
    u = (u**(1-prefs.vareps)-1.0)/(1-prefs.vareps)
    vopt = bo_fun(u,beq,prefs.beta,prefs.gamma,prefs.vareps)
    return vopt


@njit(float64(float64,float64,int64, int64, int64,float64[:,:],
              float64[:,:],float64[:],float64[:],float64[:,:,:],
              float64, set_prefs.class_type.instance_type,
              set_dims.class_type.instance_type,
              set_rates.class_type.instance_type, float64[:,:,:,:,:]),
      fastmath=True, cache=True)
def v_bo_fun(cons, x, z, t, i_hh, p_h, b_its, f_h, nu_ij_c,
            qs_ij, base_value, prefs, dims, rates, nextv):
    i_d,i_w,i_s,i_e, i_h = dims.adm[z,:]
    d0 = dims.d_h[i_d,i_e,t]
    w_t = x - cons
    if w_t >= 0.0:
        w_t *= np.exp(rates.rate)
    else :
        r_b = i_h*rates.r_h + (1.0-i_h)*rates.r_r
        w_t *= np.exp(r_b)
    d_t = i_hh*(rates.xi_d*i_h*d0
                + (1.0 - i_h)*rates.omega_d*p_h[i_e,t])
    q_ss = qs_ij[i_s,:,t]
    ev = 0.0
    for i_ee in range(dims.n_e):
        d_low, d_up, du = scale(d_t, dims.d_h[:,i_ee,t+1])
        for i_ss in range(dims.n_s):
            ww_space = dims.w_space[d_low,:,i_ss,i_ee,i_hh,t+1]
            w_low, w_up, wu = scale(w_t, ww_space)
            beq = beq_fun(d_t,w_t,i_hh,p_h[i_ee,t+1],b_its[i_ee,t+1],
                              rates.tau_s0, rates.tau_s1)
            v = nextv[:,:,i_ss,i_ee,i_hh]
            if i_ss < (dims.n_s-1):
                pv = v[d_low,w_low] + du*(-v[d_low,w_low]+v[d_up,w_low]) + \
                     wu*(-v[d_low,w_low] + v[d_low,w_up]) + \
                     wu*du*(v[d_low, w_low] - v[d_up,w_low]
                            - v[d_low,w_up] + v[d_up,w_up])
                ev += f_h[i_ee] * q_ss[i_ss] * np.exp(-prefs.gamma*pv)
            else :
                if prefs.b_x > 0.0:
                    bu = bk_fun(beq,prefs.b_x,prefs.b_k,prefs.vareps)
                    ev += f_h[i_ee]*q_ss[i_ss] *np.exp(-prefs.gamma*(1-prefs.beta)*bu)
    amen = (rates.phi + prefs.nu_h * i_h) * p_h[2, 0]
    eqscale = 1.0
    if dims.n_s==16 and dims.a_j[i_s]==1:
        eqscale += rates.eqscale
    u = cob_fun(cons, amen, nu_ij_c[i_s],prefs.vareps,  prefs.rho,eqscale)
    u = (u**(1-prefs.vareps)-1.0)/(1-prefs.vareps)
    vopt = bo_fun(u,ev,prefs.beta,prefs.gamma,prefs.vareps)
    return vopt

# create a new series of function for purchasing at a specific age, integrating purchase.
@njit(Tuple((float64[:],float64[:,:],float64[:,:]))(int64,float64[:,:],
            float64[:,:],float64[:,:],float64[:],float64[:],
            float64[:], float64[:,:], float64[:,:,:], int64, float64,
            set_dims.class_type.instance_type,
            set_rates.class_type.instance_type,
            set_prefs.class_type.instance_type,
            set_prices.class_type.instance_type,
            set_benfs.class_type.instance_type, float64[:,:,:,:,:],
            float64[:,:,:,:,:],float64[:,:],int64, int64),
            fastmath=True, parallel=False, cache=True)
def core_fun(t, p_h, p_r, b_its, f_h, nu_ij_c,  med_ij, y_ij,
             qs_ij, married, base_value, dims, rates,
             prefs, prices, benfs, nextv, nextv_ref, clast, isolve, delay_yrs = 0):
    n_c = 10
    vs = np.empty(dims.n_adm,dtype=np.float64)
    cs = np.empty((dims.n_adm,2),dtype=np.float64)
    condvs = np.empty((dims.n_adm,2),dtype=np.float64)
    vs_ = np.empty(n_c,dtype=np.float64)
    vopt = np.empty(2, dtype=np.float64)
    copt = np.empty(2, dtype=np.float64)
    for z in range(dims.n_adm):
        i_d = dims.adm[z,0]
        i_w = dims.adm[z,1]
        i_s = dims.adm[z,2]
        i_e = dims.adm[z,3]
        i_h = dims.adm[z,4]
        d0 = dims.d_h[i_d, i_e, t]
        w0 = dims.w_space[i_d, i_w, i_s, i_e, i_h, t]
        vopt[:] = 0.0
        copt[:] = 0.0
        afford = 1
        for i_hh in range(2):
            if (dims.ij_h[i_s]==0 or i_h==0) and i_hh==1:
                continue
            cash  = x_fun(d0,w0,i_h,dims.s_i[i_s],dims.s_j[i_s],married,i_hh,
                                     t, p_h[i_e,t],p_r[i_e,t],
                                     b_its[i_e,t], med_ij[i_s], y_ij[i_s,t],
                                     dims,rates, prices, benfs, delay_yrs)
            if i_h==0:
                x_w = rates.omega_r * y_ij[i_s,t]
                r_b = rates.r_r
            else :
                x_w = min(min(rates.omega_h0*p_h[i_e,t],
                          rates.omega_h1*max(p_h[i_e,t]-d0,0.0)),rates.omega_r * y_ij[i_s,t])
                r_b = rates.r_h
            c_max = x_w*np.exp(-r_b) + cash[0]
            afford = 1
            x_f = 0.0
            if dims.s_i[i_s]<=2:
                x_f += rates.x_min
            if married == 1:
                if dims.s_j[i_s]<=2:
                    if dims.s_i[i_s]>2:
                        x_f += rates.x_min
                    else :
                        x_f += rates.x_min * rates.eqscale
            if c_max <= x_f and i_hh==1:
                afford = 0
                continue
            if c_max <= x_f and i_hh==0:
                c_max = x_f
            if isolve==1:
                cs_ = np.linspace(np.sqrt(x_f), np.sqrt(c_max), n_c)
                if t==dims.t_last:
                    if c_max==x_f:
                        vs_[:] = v_t_fun(cs_[0]**2,cash[0],z,i_hh,p_h,b_its,f_h, nu_ij_c, base_value, prefs, dims, rates, delay_yrs)
                    else :
                        for i_c in range(n_c):
                            vs_[i_c] = v_t_fun(cs_[i_c]**2,cash[0],z,i_hh,p_h,b_its,f_h, nu_ij_c, base_value, prefs, dims, rates, delay_yrs)
                else :
                    if cash[2]==0:
                        vprime = nextv
                    else :
                        vprime = nextv_ref
                    if c_max==x_f:
                        vs_[:] = v_fun(cs_[0]**2,cash[0],z,t,i_hh,p_h,b_its,f_h, nu_ij_c,qs_ij, base_value, prefs, dims, rates, vprime, delay_yrs)
                    else :
                        for i_c in range(n_c):
                            vs_[i_c] = v_fun(cs_[i_c]**2,cash[0],z,t,i_hh,p_h,b_its,f_h, nu_ij_c,qs_ij, base_value, prefs, dims, rates, vprime, delay_yrs)
                imax = np.argmax(vs_)
                copt[i_hh] = cs_[imax]**2
                vopt[i_hh] = vs_[imax]
            else :
                copt[i_hh] = clast[z,i_hh]
                if cash[2]==0:
                    vprime = nextv
                else :
                    vprime = nextv_ref
                if t==dims.t_last:
                    vopt[i_hh] = v_t_fun(copt[i_hh],cash[0],z,i_hh,p_h,b_its,f_h,
                        nu_ij_c,base_value,prefs, dims, rates, delay_yrs)
                else :
                    vopt[i_hh] = v_fun(copt[i_hh],cash[0],z,t,i_hh,p_h,b_its,f_h,
                        nu_ij_c, qs_ij,base_value,prefs, dims, rates, vprime, delay_yrs)
        if i_h==1 and dims.ij_h[i_s]==1 and afford==1:
            if vopt[1] > vopt[0]:
                vs[z] = vopt[1]
            else :
                vs[z] = vopt[0]
        else :
            vs[z] = vopt[0]
            vopt[1] = -999.0
            copt[1] = copt[0]
        # option not to purchase if delaying
        cs[z,:] = copt[:]
        condvs[z,:] = vopt[:]
    return vs, cs, condvs



