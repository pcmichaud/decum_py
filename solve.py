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

def setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp, surv_bias,
                  sp_surv_bias,miss_par=0.0,sp_miss_par=0.0):
    # create rates
    rates = set_rates()
    # create dimensions
    dims = set_dims(hh['married'], rates.omega_d)
    # house price dynamics and matrices
    p_h, f_h, p_r = house_prices(g, sig, base_value, hh['home_value'], rates,
                                 dims)
    # finish up state space
    dims.set_dspace(p_h)
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
                          surv_bias['miss_psurv85'], miss_par, dims.T)
    if hh['married'] == 1:
        gammas, deltas = parse_surv(hp_sp)
        q1_sp = transition_rates(sp['sp_age'], gammas, deltas,
                                 sp_surv_bias['xi_sp'],
                                 sp_surv_bias['sp_miss_psurv85'], sp_miss_par, dims.T)
        q1_ij = joint_surv_rates(q1, q1_sp, dims.n_s, dims.T)
    else :
        q1_ij = q1[:, :, :]
    return p_h, f_h, p_r, y_ij, med_ij, q1_ij, dims, rates

def get_rules(hh, rp, sp, base_value, prices, benfs,
              p_h, f_h, p_r, y_ij, med_ij, qs_ij, b_its, nu_ij_c,
              rates, dims, prefs):
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
                base_value, dims, rates, prefs, prices, benfs, nextv, cc_last, 1)

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
        cc_last = np.copy(c_last)
        v_last, c_last, condv_last = core_fun(t, p_h, p_r,b_its,
                f_h, nu_ij_c, med_ij, y_ij, qs_ij, hh['married'],
                base_value, dims, rates, prefs, prices, benfs, nextv, cc_last, isolve)
        v_t[dims.to_states[:],t] = v_last[:]
        c_t[dims.to_states[:],0,t] = c_last[:,0]
        c_t[dims.to_states[:],1,t] = c_last[:,1]
        condv_t[dims.to_states[:],0,t] = condv_last[:,0]
        condv_t[dims.to_states[:],1,t] = condv_last[:,1]
    return c_t, condv_t

def get_value(hh, rp, sp, base_value, prices, benfs,
              p_h, f_h, p_r, y_ij, med_ij, qs_ij, b_its, nu_ij_c, 
              rates, dims, prefs):
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
                base_value, dims, rates, prefs, prices, benfs, nextv, cc_last, 1)

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
        cc_last = np.copy(c_last)
        v_last, c_last, condv_last = core_fun(t, p_h, p_r,b_its,
                f_h, nu_ij_c, med_ij, y_ij, qs_ij, hh['married'],
                base_value, dims, rates, prefs, prices, benfs, nextv, cc_last, isolve)
        v_t[dims.to_states[:],t] = v_last[:]
        c_t[dims.to_states[:],0,t] = c_last[:,0]
        c_t[dims.to_states[:],1,t] = c_last[:,1]
        condv_t[dims.to_states[:],0,t] = condv_last[:,0]
        condv_t[dims.to_states[:],1,t] = condv_last[:,1]
    # find current state
    h_init = hh['own']
    e_init = 2
    s_init = rp['hlth']-1
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
    return value

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

    # find current state
    i_h = hh['own']
    i_e = 2
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
                dims,rates, prices, benfs)
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

    return cons_path, own_path, wlth_path, home_path



@njit(float64(float64,float64,int64,int64,float64[:,:],
              float64[:,:],float64[:],float64[:],
              float64,set_prefs.class_type.instance_type,
              set_dims.class_type.instance_type,
              set_rates.class_type.instance_type),fastmath=True, cache=True)
def v_t_fun_ez(cons, x, z, i_hh, p_h, b_its, f_h, nu_ij_c,
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
            if prefs.gamma == 1.0:
                beq += prefs.b_x*f_h[i_ee]*np.log((b_e + prefs.b_k))
            else :
                beq += prefs.b_x*f_h[i_ee]*(((b_e + prefs.b_k))**(
                1.0-prefs.gamma))
    amen = i_h * p_h[2, 0]
    eqscale = 1.0
    if dims.n_s==16 and dims.a_j[i_s]==1:
        eqscale += rates.eqscale
    u = ces_fun(cons, amen, nu_ij_c[i_s], prefs.nu_h, prefs.rho, eqscale)
    beta = prefs.beta
    vareps = prefs.varepsilon
    gamma = prefs.gamma
    present = (1.0-beta)*(u**(1.0-vareps))
    if beq>0.0:
        if gamma == 1.0:
            future = beta * (np.exp(beq)**(1.0-vareps))
        else :
            future = beta * (beq ** ((1.0-vareps)/(1.0-gamma)))
    else :
        future = 0.0
    vopt = (present + future)**(1.0/(1.0-vareps))
    return vopt

@njit(float64(float64,float64,int64, int64, int64,float64[:,:],
              float64[:,:],float64[:],float64[:],float64[:,:,:],
              float64, set_prefs.class_type.instance_type,
              set_dims.class_type.instance_type,
              set_rates.class_type.instance_type, float64[:,:,:,:,:]),
      fastmath=True, cache=True)
def v_fun_ez(cons, x, z, t, i_hh, p_h, b_its, f_h, nu_ij_c,
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
            beq = 0.0
            if prefs.b_x > 0.0:
                beq = beq_fun(d_t,w_t,i_hh,p_h[i_ee,t+1],b_its[i_ee,t+1],
                              rates.tau_s0, rates.tau_s1)
            v = nextv[:,:,i_ss,i_ee,i_hh]
            if i_ss < (dims.n_s-1):
                pv = v[d_low,w_low] + du*(-v[d_low,w_low]+v[d_up,w_low]) + \
                     wu*(-v[d_low,w_low] + v[d_low,w_up]) + \
                     wu*du*(v[d_low, w_low] - v[d_up,w_low]
                            - v[d_low,w_up] + v[d_up,w_up])
                if prefs.gamma != 1.0:
                    pvv = pv ** (1.0-prefs.gamma)
                    ev += f_h[i_ee] * q_ss[i_ss] * pvv
                else :
                    ev += f_h[i_ee] * q_ss[i_ss] * np.log(pv)
            else :
                if prefs.b_x > 0.0 and beq > 0.0:
                    if prefs.gamma != 1.0:
                        pvv = prefs.b_x *(( (beq + prefs.b_k)) ** (
                                1.0 - prefs.gamma))
                        if pvv>1e6:
                            pvv = 1e6
                        ev += f_h[i_ee] * q_ss[i_ss] * pvv
                    else :
                        ev += f_h[i_ee] * q_ss[i_ss] * prefs.b_x *np.log( (
                            beq + prefs.b_k))
    amen = i_h * p_h[2,0]
    eqscale = 1.0
    if dims.n_s==16 and dims.a_j[i_s]==1:
        eqscale += rates.eqscale
    u = ces_fun(cons, amen, nu_ij_c[i_s], prefs.nu_h, prefs.rho,eqscale)
    vopt = ez_fun(u, ev, prefs.beta, prefs.varepsilon, prefs.gamma)
    return vopt

@njit(Tuple((float64[:],float64[:,:],float64[:,:]))(int64,float64[:,:],
            float64[:,:],float64[:,:],float64[:],float64[:],
            float64[:], float64[:,:], float64[:,:,:], int64, float64,
            set_dims.class_type.instance_type,
            set_rates.class_type.instance_type,
            set_prefs.class_type.instance_type,
            set_prices.class_type.instance_type,
            set_benfs.class_type.instance_type, float64[:,:,:,:,:],float64[:,:],int64),
            fastmath=True, parallel=False, cache=True)
def core_fun(t, p_h, p_r, b_its, f_h, nu_ij_c,  med_ij, y_ij,
             qs_ij, married, base_value, dims, rates,
             prefs, prices, benfs, nextv, clast, isolve):
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
            if dims.ij_h[i_s]==0 and i_hh==1:
                continue
            cash = x_fun(d0,w0,i_h,dims.s_i[i_s],dims.s_j[i_s],married,i_hh,
                         t, p_h[i_e,t],p_r[i_e,t],
                         b_its[i_e,t], med_ij[i_s], y_ij[i_s,t],
                         dims,rates, prices, benfs)
            if i_h==0:
                x_w = rates.omega_r * y_ij[i_s,t]
                r_b = rates.r_r
            else :
                x_w = min(min(rates.omega_h0*p_h[i_e,t],
                          rates.omega_h1*max(p_h[i_e,t]-d0,0.0)),rates.omega_r * y_ij[i_s,t])
                r_b = rates.r_h
            c_max = max(x_w*np.exp(-r_b) + cash[0],0.0)
            afford = 1
            if c_max <= 0.0 and i_hh==1:
                afford = 0
                continue
            if isolve==1:
                cs_ = np.linspace(np.sqrt(rates.x_min), np.sqrt(c_max), n_c)
                if t==dims.t_last:
                    for i_c in range(n_c):
                        vs_[i_c] = v_t_fun_ez(cs_[i_c]**2,cash[0],z,i_hh,p_h,b_its,f_h, nu_ij_c, base_value, prefs, dims, rates)
                else :
                    for i_c in range(n_c):
                        vs_[i_c] = v_fun_ez(cs_[i_c]**2,cash[0],z,t,i_hh,p_h,b_its,f_h, nu_ij_c,qs_ij, base_value, prefs, dims, rates, nextv)
                imax = np.argmax(vs_)
                copt[i_hh] = cs_[imax]**2
                vopt[i_hh] = vs_[imax]
            else :
                copt[i_hh] = clast[z,i_hh]
                if t==dims.t_last:
                    vopt[i_hh] = v_t_fun_ez(copt[i_hh],cash[0],z,i_hh,p_h,b_its,f_h,
                        nu_ij_c,base_value,prefs, dims, rates)
                else :
                    vopt[i_hh] = v_fun_ez(copt[i_hh],cash[0],z,t,i_hh,p_h,b_its,f_h,
                        nu_ij_c, qs_ij,base_value,prefs, dims, rates, nextv)
        if dims.ij_h[i_s]==1 and afford==1:
            if vopt[1] > vopt[0]:
                vs[z] = vopt[1]
            else :
                vs[z] = vopt[0]
        else :
            vs[z] = vopt[0]
            vopt[1] = -999.0
            copt[1] = copt[0]
        cs[z,:] = copt[:]
        condvs[z,:] = vopt[:]
    return vs, cs, condvs

