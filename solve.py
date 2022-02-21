import numpy as np
from tools import *
from actors import *
from survival import *
from prefs import *
from numba import njit, float64, int64
from numba.types import Tuple

def setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp, surv_bias,
                  sp_surv_bias,miss_par=0.0,sp_miss_par=0.0):
    # create rates
    n = 5
    rates = set_rates(n)
    # create dimensions
    dims = set_dims(hh['married'], rates.omega_d, n = n)
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
    qs_ij = adjust_surv(q1_ij, dims.time_t, dims.n_s, dims.T, dims.n)
    return p_h, f_h, p_r, y_ij, med_ij, qs_ij, dims, rates

def get_value(hh, rp, sp, base_value, prices, benfs,
              p_h, f_h, p_r, y_ij, med_ij, qs_ij, b_its, nu_ij_c, nu_ij_h,
              rates, dims, prefs):
    v_t = np.zeros((dims.n_states, dims.T),dtype='float64')
    # solve final year for admissible states
    v_last = core_t_fun(p_h, p_r, b_its,
                f_h, nu_ij_c, nu_ij_h, med_ij, y_ij, hh['married'],
                base_value, dims, rates, prefs, prices, benfs)
    # map those to all states
    v_t[dims.to_states[:],dims.t_last] = v_last[:]
    # solve remaining years
    for i in reversed(range(dims.nper-1)):
        j = dims.time_t[i]
        #print(i,j,j+dims.n)
        nextv = v_t[:,j+dims.n].reshape((dims.n_d,dims.n_w,dims.n_s,
                                          dims.n_e,2),order='F')
        v_last = core_fun(j, p_h, p_r,b_its,
                f_h, nu_ij_c, nu_ij_h, med_ij, y_ij, qs_ij, hh['married'],
                base_value, dims, rates, prefs, prices, benfs, nextv)
        v_t[dims.to_states[:],j] = v_last[:]
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
    ww_space = dims.w_space[d_low,:,s_init,e_init,h_init,0]
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

@njit(float64(float64,float64,int64,int64,float64[:,:],
              float64[:,:],float64[:],float64[:],float64[:],
              float64,set_prefs.class_type.instance_type,
              set_dims.class_type.instance_type,
              set_rates.class_type.instance_type),fastmath=True, cache=True)
def v_t_fun_ez(cons, x, z, i_hh, p_h, b_its, f_h, nu_ij_c,
            nu_ij_h,  base_value, prefs, dims, rates):
    i_d,i_w,i_s,i_e, i_h = dims.adm[z,:]
    d0 = dims.d_h[i_d,i_e,dims.t_last]
    w_t = x - cons
    if w_t >= 0.0:
        w_t *= np.exp(rates.rate)
    else :
        r_b = i_h*rates.r_h + (1.0-i_h)*rates.r_r
        w_t *= np.exp(r_b)
    d_t = i_hh*(rates.xi_d*i_h*d0
                + (1.0 - i_h)*rates.omega_d*p_h[i_e,dims.t_last])
    beq = 0.0
    if prefs.b_x>0.0:
        for i_ee in range(dims.n_e):
            b_e = beq_fun(d_t,w_t,i_hh,p_h[i_ee,dims.t_last],
                          b_its[i_ee,dims.t_last],rates.tau_s0,rates.tau_s1)
            if prefs.gamma == 1.0:
                beq += f_h[i_ee]*np.log(prefs.b_x*(b_e + prefs.b_k))
            else :
                beq += f_h[i_ee]*((prefs.b_x*(b_e + prefs.b_k))**(
                1.0-prefs.gamma))
    amen = i_hh * p_h[2, 0]
    u = ces_fun(cons, amen, nu_ij_c[i_s], nu_ij_h[i_s],prefs.rho)
    beta_n = prefs.beta**dims.n
    vareps = prefs.varepsilon
    gamma = prefs.gamma
    present = (1.0-beta_n)*(u**(1.0-vareps))
    if beq>0.0:
        if gamma == 1.0:
            future = beta_n * (np.exp(beq)**(1.0-vareps))
        else :
            future = beta_n * (beq ** ((1.0-vareps)/(1.0-gamma)))
    else :
        future = 0.0
    vopt = (present + future)**(1.0/(1.0-vareps))
    return vopt

@njit(float64(float64,float64,int64,int64,float64[:,:],
              float64[:,:],float64[:],float64[:],float64[:],
              float64,set_prefs.class_type.instance_type,
              set_dims.class_type.instance_type,
              set_rates.class_type.instance_type),fastmath=True, cache=True)
def v_t_fun_eu(cons, x, z, i_hh, p_h, b_its, f_h, nu_ij_c,
            nu_ij_h,  base_value, prefs, dims, rates):
    i_d,i_w,i_s,i_e, i_h = dims.adm[z,:]
    d0 = dims.d_h[i_d,i_e,dims.t_last]
    w_t = x - cons
    if w_t >= 0.0:
        w_t *= np.exp(rates.rate)
    else :
        r_b = i_h*rates.r_h + (1.0-i_h)*rates.r_r
        w_t *= np.exp(r_b)
    d_t = i_hh*(rates.xi_d*i_h*d0
                + (1.0 - i_h)*rates.omega_d*p_h[i_e,dims.t_last])
    beq = 0.0
    if prefs.b_x>0.0:
        for i_ee in range(dims.n_e):
            b_e = beq_fun(d_t,w_t,i_hh,p_h[i_ee,dims.t_last],
                          b_its[i_ee,dims.t_last],rates.tau_s0,rates.tau_s1)
            beq += f_h[i_ee]* prefs.b_x* ((b_e + prefs.b_k)**(
                1.0-prefs.gamma))/(1.0-prefs.gamma)
    amen = i_hh * p_h[2, 0]
    u = ces_fun(cons, amen, nu_ij_c[i_s], nu_ij_h[i_s],prefs.rho)
    beta_n = prefs.beta**dims.n
    vareps = prefs.varepsilon
    gamma = prefs.gamma
    present = u**(1.0-gamma)/(1.0-gamma)
    if beq>0.0:
        future = beta_n * beq
    else :
        future = 0.0
    vopt = present + future
    return vopt

@njit(float64[:](float64[:,:],
            float64[:,:],float64[:,:],float64[:],float64[:],
            float64[:],float64[:], float64[:,:],int64, float64,
            set_dims.class_type.instance_type,
            set_rates.class_type.instance_type,
            set_prefs.class_type.instance_type,
            set_prices.class_type.instance_type,
            set_benfs.class_type.instance_type), fastmath=True,
            parallel=False, cache=True)
def core_t_fun(p_h, p_r, b_its, f_h, nu_ij_c, nu_ij_h,  med_ij, y_ij, married,
               base_value, dims, rates, prefs, prices, benfs):
    tol = 1e-3
    r = 0.61803399
    c = 1-r
    vs = np.empty(dims.n_adm,dtype=np.float64)
    vopt = np.empty(2, dtype=np.float64)
    for z in range(dims.n_adm):
        i_d = dims.adm[z,0]
        i_w = dims.adm[z,1]
        i_s = dims.adm[z,2]
        i_e = dims.adm[z,3]
        i_h = dims.adm[z,4]
        t_last = dims.t_last
        d0 = dims.d_h[i_d, i_e, t_last]
        w0 = dims.w_space[i_d, i_w, i_s, i_e, i_h, t_last]
        vopt[:] = 0.0
        for i_hh in range(2):
            if dims.ij_h[i_s]==0 and i_hh==1:
                continue
            cash = x_fun(d0,w0,i_h,dims.s_i[i_s],dims.s_j[i_s],married,i_hh,
                         t_last, p_h[i_e,t_last],p_r[i_e,t_last],
                         b_its[i_e,t_last], med_ij[i_s],y_ij[i_s,t_last],
                         dims,rates,prices, benfs)
            if i_h==0:
                x_w = rates.omega_r * y_ij[i_s,t_last]
                r_b = rates.r_r
            else :
                x_w = min(rates.omega_h0*p_h[i_e,t_last],
                          rates.omega_h1*max(p_h[i_e,t_last]-d0,0.0))
                r_b = rates.r_h
            c_max = max(x_w*np.exp(-r_b) + cash[0],0.0)
            afford = 1
            if c_max <= 0.0 and i_hh==1:
                afford = 0
                continue
            ax = 0.25*c_max
            bx = 0.5*c_max
            cx = c_max
            x0 = ax
            x3 = cx
            if (np.abs(cx-bx)>np.abs(bx-ax)):
                x1 = bx
                x2 = bx + c*(cx - bx)
            else :
                x2 = bx
                x1 = bx - c*(bx - ax)

            f1  = -v_t_fun_ez(x1, cash[0], z, i_hh, p_h, b_its, f_h,
                          nu_ij_c, nu_ij_h,  base_value, prefs, dims, rates)
            f2  = -v_t_fun_ez(x2, cash[0], z, i_hh, p_h, b_its, f_h,
                          nu_ij_c, nu_ij_h,  base_value, prefs, dims, rates)
            while (np.abs(x3-x0) > tol*(np.abs(x1)+np.abs(x2))):
                if f2 < f1:
                    x0 = x1
                    x1 = x2
                    x2 = r*x1 + c*x3
                    f0 = f1
                    f1 = f2
                    f2  = -v_t_fun_ez(x2, cash[0], z, i_hh, p_h, b_its, f_h,
                          nu_ij_c, nu_ij_h,  base_value, prefs, dims, rates)
                else :
                    x3 = x2
                    x2 = x1
                    x1 = r*x2 + c*x0
                    f3 = f2
                    f2 = f1
                    f1  = -v_t_fun_ez(x1, cash[0], z, i_hh, p_h, b_its, f_h,
                          nu_ij_c, nu_ij_h,  base_value, prefs, dims, rates)
            if f1<f2:
                vopt[i_hh] = -f1
            else :
                vopt[i_hh] = -f2
        if dims.ij_h[i_s]==1 and afford==1:
            if vopt[1] > vopt[0]:
                vs[z] = vopt[1]
            else :
                vs[z] = vopt[0]
        else :
            vs[z] = vopt[0]
    return vs

@njit(float64(float64,float64,int64, int64, int64,float64[:,:],
              float64[:,:],float64[:],float64[:],float64[:], float64[:,:,:],
              float64, set_prefs.class_type.instance_type,
              set_dims.class_type.instance_type,
              set_rates.class_type.instance_type, float64[:,:,:,:,:]),
      fastmath=True, cache=True)
def v_fun_ez(cons, x, z, t, i_hh, p_h, b_its, f_h, nu_ij_c,
            nu_ij_h,  qs_ij, base_value, prefs, dims, rates, nextv):
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
        d_low, d_up, du = scale(d_t, dims.d_h[:,i_ee,t+dims.n])
        for i_ss in range(dims.n_s):
            ww_space = dims.w_space[d_low,:,i_ss,i_ee,i_hh,t+dims.n]
            w_low, w_up, wu = scale(w_t, ww_space)
            beq = 0.0
            if prefs.b_x > 0.0:
                beq = beq_fun(d_t,w_t,i_hh,p_h[i_ee,t+dims.n],b_its[i_ee,t+dims.n],
                              rates.tau_s0, rates.tau_s1)
            v = nextv[:,:,i_ss,i_ee,i_hh]
            if i_ss < (dims.n_s-1):
                pv = v[d_low,w_low] + du*(-v[d_low,w_low]+v[d_up,w_low]) + \
                     wu*(-v[d_low,w_low] + v[d_low,w_up]) + \
                     wu*du*(v[d_low, w_low] - v[d_up,w_low]
                            - v[d_low,w_up] + v[d_up,w_up])
                if prefs.gamma != 1.0:
                    pvv = pv ** (1.0-prefs.gamma)
                    #if pvv>1e6:
                    #    pvv = 1e6
                    ev += f_h[i_ee] * q_ss[i_ss] * pvv
                else :
                    ev += f_h[i_ee] * q_ss[i_ss] * np.log(pv)
            else :
                if prefs.b_x > 0.0 and beq > 0.0:
                    if prefs.gamma != 1.0:
                        pvv = ((prefs.b_x * (beq + prefs.b_k)) ** (
                                1.0 - prefs.gamma))
                        if pvv>1e6:
                            pvv = 1e6
                        ev += f_h[i_ee] * q_ss[i_ss] * pvv
                    else :
                        ev += f_h[i_ee] * q_ss[i_ss] * np.log(prefs.b_x * (
                            beq + prefs.b_k))
    amen = i_hh * p_h[2,0]
    u = ces_fun(cons, amen, nu_ij_c[i_s], nu_ij_h[i_s],prefs.rho)
    vopt = ez_fun(u, ev, prefs.beta, prefs.varepsilon, prefs.gamma, dims.n)
    return vopt

@njit(float64(float64,float64,int64, int64, int64,float64[:,:],
              float64[:,:],float64[:],float64[:],float64[:], float64[:,:,:],
              float64, set_prefs.class_type.instance_type,
              set_dims.class_type.instance_type,
              set_rates.class_type.instance_type, float64[:,:,:,:,:]),
      fastmath=True, cache=True)
def v_fun_eu(cons, x, z, t, i_hh, p_h, b_its, f_h, nu_ij_c,
            nu_ij_h,  qs_ij, base_value, prefs, dims, rates, nextv):
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
        d_low, d_up, du = scale(d_t, dims.d_h[:,i_ee,t+dims.n])
        for i_ss in range(dims.n_s):
            ww_space = dims.w_space[d_low,:,i_ss,i_ee,i_hh,t+dims.n]
            w_low, w_up, wu = scale(w_t, ww_space)
            v = nextv[:,:,i_ss,i_ee,i_hh]
            if i_ss < (dims.n_s-1):
                pv = v[d_low,w_low] + du*(-v[d_low,w_low]+v[d_up,w_low]) + \
                     wu*(-v[d_low,w_low] + v[d_low,w_up]) + \
                     wu*du*(v[d_low, w_low] - v[d_up,w_low]
                            - v[d_low,w_up] + v[d_up,w_up])
                ev += f_h[i_ee] * q_ss[i_ss] * pv
            else :
                beq = 0.0
                if prefs.b_x > 0.0:
                    beq = beq_fun(d_t, w_t, i_hh, p_h[i_ee, t + dims.n],
                                  b_its[i_ee, t + dims.n],
                                  rates.tau_s0, rates.tau_s1)
                    pv = prefs.b_x*((beq + prefs.b_k) ** (
                                1.0 - prefs.gamma))/(1.0-prefs.gamma)
                    ev += f_h[i_ee] * q_ss[i_ss] * pv
    amen = i_hh * p_h[2,0]
    u = ces_fun(cons, amen, nu_ij_c[i_s], nu_ij_h[i_s],prefs.rho)
    vopt = eu_fun(u, ev, prefs.beta, prefs.varepsilon, prefs.gamma, dims.n)
    return vopt

@njit(float64[:](int64,float64[:,:],
            float64[:,:],float64[:,:],float64[:],float64[:],
            float64[:],float64[:], float64[:,:], float64[:,:,:], int64, float64,
            set_dims.class_type.instance_type,
            set_rates.class_type.instance_type,
            set_prefs.class_type.instance_type,
            set_prices.class_type.instance_type,
            set_benfs.class_type.instance_type, float64[:,:,:,:,:]),
            fastmath=True, parallel=False, cache=True)
def core_fun(t, p_h, p_r, b_its, f_h, nu_ij_c, nu_ij_h,  med_ij, y_ij,
             qs_ij, married, base_value, dims, rates,
             prefs, prices, benfs, nextv):
    tol = 1e-3
    r = 0.61803399
    c = 1-r
    vs = np.empty(dims.n_adm,dtype=np.float64)
    vopt = np.empty(2, dtype=np.float64)
    for z in range(dims.n_adm):
        i_d = dims.adm[z,0]
        i_w = dims.adm[z,1]
        i_s = dims.adm[z,2]
        i_e = dims.adm[z,3]
        i_h = dims.adm[z,4]
        d0 = dims.d_h[i_d, i_e, t]
        w0 = dims.w_space[i_d, i_w, i_s, i_e, i_h, t]
        vopt[:] = 0.0
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
                x_w = min(rates.omega_h0*p_h[i_e,t],
                          rates.omega_h1*max(p_h[i_e,t]-d0,0.0))
                r_b = rates.r_h
            c_max = max(x_w*np.exp(-r_b) + cash[0],0.0)
            afford = 1
            if c_max <= 0.0 and i_hh==1:
                afford = 0
                continue
            ax = 0.25*c_max
            bx = 0.5*c_max
            cx = c_max
            x0 = ax
            x3 = cx
            if (np.abs(cx-bx)>np.abs(bx-ax)):
                x1 = bx
                x2 = bx + c*(cx - bx)
            else :
                x2 = bx
                x1 = bx - c*(bx - ax)

            f1  = -v_fun_ez(x1, cash[0], z, t,  i_hh, p_h, b_its, f_h,
                          nu_ij_c, nu_ij_h, qs_ij,  base_value, prefs, dims,
                                                  rates,
                         nextv)
            f2  = -v_fun_ez(x2, cash[0], z, t,  i_hh, p_h, b_its, f_h,
                          nu_ij_c, nu_ij_h, qs_ij, base_value, prefs, dims,
                                                  rates,
                         nextv)
            while (np.abs(x3-x0) > tol*(np.abs(x1)+np.abs(x2))):
                if f2 < f1:
                    x0 = x1
                    x1 = x2
                    x2 = r*x1 + c*x3
                    f0 = f1
                    f1 = f2
                    f2  = -v_fun_ez(x2, cash[0], z, t,  i_hh, p_h, b_its, f_h,
                          nu_ij_c, nu_ij_h,  qs_ij, base_value, prefs, dims,
                                 rates,
                                 nextv)
                else :
                    x3 = x2
                    x2 = x1
                    x1 = r*x2 + c*x0
                    f3 = f2
                    f2 = f1
                    f1  = -v_fun_ez(x1, cash[0], z, t, i_hh, p_h, b_its, f_h,
                          nu_ij_c, nu_ij_h,  qs_ij, base_value, prefs, dims,
                                 rates,
                                 nextv)
            if f1<f2:
                vopt[i_hh] = -f1
            else :
                vopt[i_hh] = -f2
        if dims.ij_h[i_s]==1 and afford==1:
            if vopt[1] > vopt[0]:
                vs[z] = vopt[1]
            else :
                vs[z] = vopt[0]
        else :
            vs[z] = vopt[0]
    return vs
