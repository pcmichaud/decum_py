from solve import *
import multiprocessing as mp
from functools import partial
from scipy.interpolate import interp1d
from itertools import product

def func_solve(row,theta):
    hh = dict(row[['wgt','cma','married','own','wealth_total',
                   'mort_balance','home_value']])
    rp = dict(row[['age','totinc','retinc','hlth']])
    if hh['married'] == 1:
        sp = dict(row[['sp_age','sp_totinc','sp_retinc','sp_hlth']])
    else:
        sp = None
    g = row['g'] * row['mu']
    sig = row['sig'] * row['zeta']
    base_value = row['base_value']
    hc = row[['hc_0','hc_1','hc_2']].to_numpy(dtype='float64')
    nh = row[['nh_0','nh_1','nh_2']].to_numpy(dtype='float64')
    hp_vars = ['gamma(2,1)', 'delta(1,2)', 'delta(2,2)', 'delta(3,2)',
               'gamma(3,1)', 'delta(1,3)', 'delta(2,3)', 'delta(3,3)',
               'gamma(4,1)', 'delta(1,4)', 'delta(2,4)', 'delta(3,4)']
    hp = row[hp_vars]
    surv_bias = row[['xi','miss_psurv85']]
    if hh['married'] == 1:
        hp_sp_vars = ['sp_' + x for x in hp_vars]
        hp_sp = row[hp_sp_vars]
        hp_sp.index = hp_vars
        sp_surv_bias = row[['xi_sp','sp_miss_psurv85']]
    else:
        hp_sp = None
        sp_surv_bias = None

    if theta is None:
        prefs = set_prefs(live_fast=row['pref_live_fast'],risk_averse=row['pref_risk_averse'],beq_money=row['pref_beq_money'],pref_home=row[
        'pref_home'])
        p_h, f_h, p_r, y_ij, med_ij, qs_ij, dims, rates = \
            setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp,
                          surv_bias,sp_surv_bias,miss_par=-0.779,sp_miss_par=0.659)
    else :
        prefs = set_prefs(varepsilon=theta[0],d_varepsilon=theta[1],
        gamma=theta[2],d_gamma=theta[3],rho=theta[4],b_x=theta[5],d_b_x=theta[6],b_k=theta[7],nu_c1=theta[8],nu_c2=theta[9],nu_h=theta[10],d_nu_h=theta[11],live_fast=row['pref_live_fast'],risk_averse=row['pref_risk_averse'],beq_money=row['pref_beq_money'],pref_home=row['pref_home'])
        p_h, f_h, p_r, y_ij, med_ij, qs_ij, dims, rates = \
            setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp,
                      surv_bias,sp_surv_bias,miss_par=theta[12],sp_miss_par=theta[13])
    nu_ij_c = update_nus(hh['married'], dims.s_i, dims.s_j, dims, prefs, rates.eqscale)
    for i in range(13):
        f_prices, f_benfs = set_scenario(row,i)
        i_prices = set_prices(f_prices[0], f_prices[1], f_prices[2])
        i_benfs = set_benfs(f_benfs[0], f_benfs[1], f_benfs[2])
        b_its = reimburse_loan(i_benfs, i_prices, p_h, dims, rates)
        row['value_' + str(i)] = get_value(hh, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                         p_r, y_ij, med_ij, qs_ij, b_its,
                                         nu_ij_c, rates, dims, prefs)
    return row

def func_simulate(row,theta):
    hh = dict(row[['wgt','cma','married','own','wealth_total',
                   'mort_balance','home_value']])
    rp = dict(row[['age','totinc','retinc','hlth']])
    if hh['married'] == 1:
        sp = dict(row[['sp_age','sp_totinc','sp_retinc','sp_hlth']])
    else:
        sp = None
    g = row['g'] * row['mu']
    sig = row['sig'] * row['zeta']
    base_value = row['base_value']
    hc = row[['hc_0','hc_1','hc_2']].to_numpy(dtype='float64')
    nh = row[['nh_0','nh_1','nh_2']].to_numpy(dtype='float64')
    hp_vars = ['gamma(2,1)', 'delta(1,2)', 'delta(2,2)', 'delta(3,2)',
               'gamma(3,1)', 'delta(1,3)', 'delta(2,3)', 'delta(3,3)',
               'gamma(4,1)', 'delta(1,4)', 'delta(2,4)', 'delta(3,4)']
    hp = row[hp_vars]
    surv_bias = row[['xi','miss_psurv85']]
    if hh['married'] == 1:
        hp_sp_vars = ['sp_' + x for x in hp_vars]
        hp_sp = row[hp_sp_vars]
        hp_sp.index = hp_vars
        sp_surv_bias = row[['xi_sp','sp_miss_psurv85']]
    else:
        hp_sp = None
        sp_surv_bias = None
    if theta is None:
        prefs = set_prefs(live_fast=row['pref_live_fast'],risk_averse=row['pref_risk_averse'],beq_money=row['pref_beq_money'],pref_home=row[
        'pref_home'])
        p_h, f_h, p_r, y_ij, med_ij, qs_ij, dims, rates = \
            setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp,
                      surv_bias,sp_surv_bias,miss_par=-0.779,sp_miss_par=0.659)
    else :
        prefs = set_prefs(varepsilon=theta[0],d_varepsilon=theta[1],
        gamma=theta[2],d_gamma=theta[3],rho=theta[4],b_x=theta[5],d_b_x=theta[6],b_k=theta[7],nu_c1=theta[8],nu_c2=theta[9],nu_h=theta[10],d_nu_h=theta[11],live_fast=row['pref_live_fast'],risk_averse=row['pref_risk_averse'],beq_money=row['pref_beq_money'],pref_home=row['pref_home'])
        p_h, f_h, p_r, y_ij, med_ij, qs_ij, dims, rates = \
            setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp,
                      surv_bias,sp_surv_bias,miss_par=theta[12],sp_miss_par=theta[13])
    nu_ij_c = update_nus(hh['married'], dims.s_i, dims.s_j, dims, prefs,rates.eqscale)

    # first solve
    i_prices = set_prices(0.0, 0.0, 0.0)
    i_benfs = set_benfs(0.0, 0.0, 0.0)
    b_its = reimburse_loan(i_benfs, i_prices, p_h, dims, rates)
    cons_rules, cond_values = get_rules(hh, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                         p_r, y_ij, med_ij, qs_ij, b_its,
                                         nu_ij_c,rates, dims, prefs)
    #print(np.mean(cons_rules[:,1,0]),np.mean(cons_rules[:,1,dims.t_last]))
    # debias survival and home prices
    # health transitions
    gammas, deltas = parse_surv(hp)
    q1 = transition_rates(rp['age'], gammas, deltas, 0.0,
                          surv_bias['miss_psurv85'], 0.0, dims.T)
    if hh['married'] == 1:
        gammas, deltas = parse_surv(hp_sp)
        q1_sp = transition_rates(sp['sp_age'], gammas, deltas,
                                 0.0,
                                 sp_surv_bias['sp_miss_psurv85'], 0.0, dims.T)
        qs_ij = joint_surv_rates(q1, q1_sp, dims.n_s, dims.T)
    else :
        qs_ij = q1[:, :, :]
    # house prices
    #p_h, f_h, p_r = house_prices(row['g'], row['sig'], base_value, hh['home_value'], rates,
    #                             dims)
    #b_its = reimburse_loan(i_benfs, i_prices, p_h, dims, rates)

    cons_path, own_path, wlth_path, home_path     = get_sim_path(row['seed'],cons_rules, cond_values, hh, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                         p_r, y_ij, med_ij, qs_ij, b_its,
                                         nu_ij_c,rates, dims, prefs)
    row[['cons_' + str(i) for i in range(dims.T)]] = cons_path
    row[['own_' + str(i) for i in range(dims.T)]] = own_path
    row[['wlth_' + str(i) for i in range(dims.T)]] = wlth_path
    row[['home_' + str(i) for i in range(dims.T)]] = home_path
    return row

def func_fair(row,theta):
    hh = dict(row[['wgt','cma','married','own','wealth_total',
                   'mort_balance','home_value']])
    rp = dict(row[['age','totinc','retinc','hlth']])
    if hh['married'] == 1:
        sp = dict(row[['sp_age','sp_totinc','sp_retinc','sp_hlth']])
    else:
        sp = None
    g = row['g'] * row['mu']
    sig = row['sig'] * row['zeta']
    base_value = row['base_value']
    hc = row[['hc_0','hc_1','hc_2']].to_numpy(dtype='float64')
    nh = row[['nh_0','nh_1','nh_2']].to_numpy(dtype='float64')
    hp_vars = ['gamma(2,1)', 'delta(1,2)', 'delta(2,2)', 'delta(3,2)',
               'gamma(3,1)', 'delta(1,3)', 'delta(2,3)', 'delta(3,3)',
               'gamma(4,1)', 'delta(1,4)', 'delta(2,4)', 'delta(3,4)']
    hp = row[hp_vars]
    surv_bias = row[['xi','miss_psurv85']]
    if hh['married'] == 1:
        hp_sp_vars = ['sp_' + x for x in hp_vars]
        hp_sp = row[hp_sp_vars]
        hp_sp.index = hp_vars
        sp_surv_bias = row[['xi_sp','sp_miss_psurv85']]
    else:
        hp_sp = None
        sp_surv_bias = None

    if theta is None:
        prefs = set_prefs(live_fast=row['pref_live_fast'],risk_averse=row['pref_risk_averse'],beq_money=row['pref_beq_money'],pref_home=row[
        'pref_home'])
        p_h, f_h, p_r, y_ij, med_ij, qs_ij, dims, rates = \
            setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp,
                          surv_bias,sp_surv_bias,miss_par=0.0,sp_miss_par=0.0)
    else :
        prefs = set_prefs(varepsilon=theta[0],d_varepsilon=theta[1],
        gamma=theta[2],d_gamma=theta[3],rho=theta[4],b_x=theta[5],d_b_x=theta[6],b_k=theta[7],nu_c1=theta[8],nu_c2=theta[9],nu_h=theta[10],d_nu_h=theta[11],live_fast=row['pref_live_fast'],risk_averse=row['pref_risk_averse'],beq_money=row['pref_beq_money'],pref_home=row['pref_home'])
        p_h, f_h, p_r, y_ij, med_ij, qs_ij, dims, rates = \
            setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp,
                      surv_bias,sp_surv_bias,miss_par=theta[12],sp_miss_par=theta[13])
    nu_ij_c = update_nus(hh['married'], dims.s_i, dims.s_j, dims, prefs, rates.eqscale)

    # objective health probabilities
    gammas, deltas = parse_surv(hp)
    q1_o = transition_rates(rp['age'], gammas, deltas, 0.0,
                          surv_bias['miss_psurv85'], 0.0, dims.T)
    if hh['married'] == 1:
        gammas, deltas = parse_surv(hp_sp)
        q1_sp_o = transition_rates(sp['sp_age'], gammas, deltas,
                                 0.0,
                                 sp_surv_bias['sp_miss_psurv85'], 0.0, dims.T)
        qs_ij_o = joint_surv_rates(q1_o, q1_sp_o, dims.n_s, dims.T)
    else :
        qs_ij_o = q1_o[:, :, :]

    # compute cumulative probabilities for health state
    q_n = np.zeros((dims.n_s,dims.T),dtype=np.float64)
    if hh['married'] == 1:
        i_s_0 = (rp['hlth']-1)*4 + sp['sp_hlth']-1
    else :
        i_s_0 = rp['hlth']-1
    q_n[i_s_0,0] = 1.0
    for i in range(1,dims.T):
        q_n[:,i] = np.dot(qs_ij_o[:,:,i-1].transpose(), q_n[:,i-1])
    # compute fair price of annuities  (see eq. 2 of paper)
    sx = np.zeros(dims.T, dtype=np.float64)
    for i in range(dims.T):
        sx[i] = np.sum(q_n[dims.s_i<=2,i])
    p_a = np.sum([np.exp(-rates.rate*i)*sx[i] for i in range(dims.T)])
    row['price_ann_fair'] = p_a
    # compute fair price of long-term care insurance (see eq. 3 of paper)
    Lx = np.zeros(dims.T, dtype=np.float64)
    qx = np.zeros(dims.T, dtype=np.float64)
    for i in range(dims.T):
        Lx[i] = np.sum(q_n[dims.s_i==2,i])
        qx[i] = np.sum(q_n[dims.s_i<=1,i])
    num = np.sum([np.exp(-rates.rate*i)*Lx[i] for i in range(dims.T)])
    den = np.sum([np.exp(-rates.rate*i)*qx[i] for i in range(dims.T)])
    p_l = num/den
    row['price_ltci_fair'] = p_l
    # find out how much want to buy, set bounds
    opt_max = np.zeros(3,dtype=np.float64)
    opt_min = np.zeros(3,dtype=np.float64)
    opt_max[0] = hh['wealth_total']
    opt_max[1] = med_ij[2]
    #rates.omega_rm = 1.0
    opt_max[2] = rates.omega_rm*(hh['home_value'] - hh['mort_balance'])
    row['max_ann_fair'] = opt_max[0]
    row['max_ltci_fair'] = opt_max[1]
    row['max_rmr_fair'] = max(opt_max[2],0.0)

    # everything is parametrized to fit into unit cube for demand
    pi = 0.5*np.ones(3,dtype=np.float64)
    pi[2] = 1.0
    i_rs = np.linspace(0.0,0.05,5)
    ds = []
    for i_r in i_rs:
        i_prices = set_prices(pi[0]*opt_max[0],p_l*pi[1]*opt_max[1] , rates.r_h + i_r)
        i_benfs = set_benfs(pi[0]*opt_max[0]/p_a, pi[1]*opt_max[1], pi[2]*opt_max[2])
        b_its = reimburse_loan(i_benfs, i_prices, p_h, dims, rates)
        # get decision rules based on fair pricing for annuities and LTCI, candidate rmr
        cons_rules, cond_values = get_rules(hh, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                            p_r, y_ij, med_ij, qs_ij, b_its,
                                            nu_ij_c,rates, dims, prefs)
        # simulate to get the house ownership dynamics
        # house prices
        p_h, f_h, p_r = house_prices(row['g'], row['sig'], base_value, hh['home_value'], rates,
                                    dims)
        b_its = reimburse_loan(i_benfs, i_prices, p_h, dims, rates)
        # find out probability that still owns original house using 500 simulations
        # keep disposition price of house
        nsim = 500
        mip = 0.0
        nneg = 0.0
        for i in range(nsim):
            cons_path, own_path, wlth_path, home_path     = get_sim_path(1234,
                                        cons_rules,cond_values, hh, rp, sp, base_value, i_prices, i_benfs, p_h, f_h, p_r, y_ij, med_ij, qs_ij, b_its,
                                        nu_ij_c,rates, dims, prefs)
            for t in range(dims.T):
                Lt = i_benfs.rmr*np.exp((rates.r_h+i_r)*t)
                lt = max(Lt - home_path[t],0)
                if own_path[t]==1:
                    mip += i_r*np.exp(-rates.r_h*t)*Lt
                if own_path[t]==0:
                    nneg += np.exp(-rates.r_h*t)*lt
                    break
        mip = mip/nsim
        nneg = nneg/nsim
        ds.append(mip-nneg)
    # interpolate the inverse function to get the fair price
    if np.sum(ds)!=0.0:
        f = interp1d(ds,i_rs,bounds_error=False)
        i_r_fair = max(f(0.0),0.0)
    else :
        i_r_fair = 0.0
    row['price_rmr_fair'] = i_r_fair
    return row


def func_joint(row,theta):
    hh = dict(row[['wgt','cma','married','own','wealth_total',
                   'mort_balance','home_value']])
    rp = dict(row[['age','totinc','retinc','hlth']])
    if hh['married'] == 1:
        sp = dict(row[['sp_age','sp_totinc','sp_retinc','sp_hlth']])
    else:
        sp = None
    g = row['g'] * row['mu']
    sig = row['sig'] * row['zeta']
    base_value = row['base_value']
    hc = row[['hc_0','hc_1','hc_2']].to_numpy(dtype='float64')
    nh = row[['nh_0','nh_1','nh_2']].to_numpy(dtype='float64')
    hp_vars = ['gamma(2,1)', 'delta(1,2)', 'delta(2,2)', 'delta(3,2)',
               'gamma(3,1)', 'delta(1,3)', 'delta(2,3)', 'delta(3,3)',
               'gamma(4,1)', 'delta(1,4)', 'delta(2,4)', 'delta(3,4)']
    hp = row[hp_vars]
    surv_bias = row[['xi','miss_psurv85']]
    if hh['married'] == 1:
        hp_sp_vars = ['sp_' + x for x in hp_vars]
        hp_sp = row[hp_sp_vars]
        hp_sp.index = hp_vars
        sp_surv_bias = row[['xi_sp','sp_miss_psurv85']]
    else:
        hp_sp = None
        sp_surv_bias = None

    if theta is None:
        prefs = set_prefs(live_fast=row['pref_live_fast'],risk_averse=row['pref_risk_averse'],beq_money=row['pref_beq_money'],pref_home=row[
        'pref_home'])
        p_h, f_h, p_r, y_ij, med_ij, qs_ij, dims, rates = \
            setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp,
                          surv_bias,sp_surv_bias,miss_par=0.0,sp_miss_par=0.0)
    else :
        prefs = set_prefs(varepsilon=theta[0],d_varepsilon=theta[1],
        gamma=theta[2],d_gamma=theta[3],rho=theta[4],b_x=theta[5],d_b_x=theta[6],b_k=theta[7],nu_c1=theta[8],nu_c2=theta[9],nu_h=theta[10],d_nu_h=theta[11],live_fast=row['pref_live_fast'],risk_averse=row['pref_risk_averse'],beq_money=row['pref_beq_money'],pref_home=row['pref_home'])
        p_h, f_h, p_r, y_ij, med_ij, qs_ij, dims, rates = \
            setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp,
                      surv_bias,sp_surv_bias,miss_par=theta[12],sp_miss_par=theta[13])
    nu_ij_c = update_nus(hh['married'], dims.s_i, dims.s_j, dims, prefs, rates.eqscale)

    # fair prices
    p_a = row['price_ann_fair_avg']
    p_l = row['price_ltci_fair_avg']
    i_r = rates.r_h + row['price_rmr_fair_avg']

    # set bounds
    opt_max = np.zeros(3,dtype=np.float64)
    opt_min = np.zeros(3,dtype=np.float64)
    opt_max[0] = hh['wealth_total']
    opt_max[1] = med_ij[2]
    rates.omega_rm = 0.35
    opt_max[2] = max(rates.omega_rm*(hh['home_value'] - hh['mort_balance']),0.0)
    # search over grid
    nu = 2
    nuu = nu*nu*nu
    gridu = np.linspace(0,1.0,nu)
    buy_grid = np.array(list(product(*[gridu,gridu,gridu])))
    #print(buy_grid)
    values = np.zeros(nuu)
    for i in range(nuu):
        pi = buy_grid[i,:]
        i_prices = set_prices(pi[0]*opt_max[0],p_l*pi[1]*opt_max[1] ,pi[2]* i_r)
        i_benfs = set_benfs(pi[0]*opt_max[0]/p_a, pi[1]*opt_max[1], pi[2]*opt_max[2])
        b_its = reimburse_loan(i_benfs, i_prices, p_h, dims, rates)
        # get decision rules based on fair pricing for annuities and LTCI, candidate rmr
        values[i] =  get_value(hh, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                         p_r, y_ij, med_ij, qs_ij, b_its,
                                         nu_ij_c, rates, dims, prefs)
    # joint purchase
    imax = np.argmax(values)
    opt_buy = buy_grid[imax,:]
    row['buy_ann_joint'] = opt_buy[0]
    row['buy_ltci_joint'] = opt_buy[1]
    row['buy_rmr_joint'] = opt_buy[2]
    # individual purchase
    i_ann = []
    i_ltci = []
    i_rmr = []
    for i in range(nuu):
        if buy_grid[i,1]==0.0 and buy_grid[i,2]==0.0:
            i_ann.append(i)
        if buy_grid[i,0]==0.0 and buy_grid[i,2]==0.0:
            i_ltci.append(i)
        if buy_grid[i,0]==0.0 and buy_grid[i,1]==0.0:
            i_rmr.append(i)
    imax = i_ann[np.argmax(values[i_ann])]
    row['buy_ann_indp'] = buy_grid[imax,0]
    imax = i_ltci[np.argmax(values[i_ltci])]
    row['buy_ltci_indp'] = buy_grid[imax,1]
    imax = i_rmr[np.argmax(values[i_rmr])]
    row['buy_rmr_indp'] = buy_grid[imax,2]
    return row


def set_scenario(row,scn):
    _prices = np.zeros(3)
    _benfs = np.zeros(3)
    if scn>=1 and scn<=4:
        _prices = np.zeros(3)
        _prices[0] = row['prem_scn_ann_'+str(scn)]
        _benfs = np.zeros(3)
        _benfs[0] = row['ben_scn_ann_'+str(scn)]
    elif scn>=5 and scn<=8:
        _prices = np.zeros(3)
        _prices[1] = row['prem_scn_ltci_'+str(scn-4)]
        _benfs = np.zeros(3)
        _benfs[1] = row['ben_scn_ltci_'+str(scn-4)]
    elif scn>=9 and scn<=12:
        _prices = np.zeros(3)
        _prices[2] = row['int_scn_rmr_'+str(scn-8)]
        _benfs = np.zeros(3)
        _benfs[2] = row['loan_scn_rmr_'+str(scn-8)]
    return _prices, _benfs


def compute_solve_chunks(df,theta):
    df = df.apply(func_solve,axis=1,args=(theta,))
    return df

def compute_fair_chunks(df,theta):
    df = df.apply(func_fair,axis=1,args=(theta,))
    return df

def compute_joint_chunks(df,theta):
    df = df.apply(func_joint,axis=1,args=(theta,))
    return df

def compute_sim_chunks(df,theta):
    df = df.apply(func_simulate,axis=1,args=(theta,))
    return df

def solve_df(data, npartitions=12,theta=None):
    # load data
    df = data.loc[:,:]
    df[['value_'+str(x) for x in range(13)]] = np.nan
    list_df = np.array_split(df, npartitions)
    compress_compute_chunks = partial(compute_solve_chunks,theta=theta)
    with mp.Pool(processes=npartitions) as pool:
        res = pool.map(compress_compute_chunks, list_df)
    df = pd.concat(res)
    return df

def solve_fair(data, npartitions=12,theta=None):
    # load data
    df = data.loc[:,:]
    df[['price_'+ p + '_fair' for p in ['ann','ltci','rmr']]] = np.nan
    df['seed'] = np.random.shuffle(df.index.to_numpy())
    list_df = np.array_split(df, npartitions)
    compress_compute_chunks = partial(compute_fair_chunks,theta=theta)
    with mp.Pool(processes=npartitions) as pool:
        res = pool.map(compress_compute_chunks, list_df)
    df = pd.concat(res)
    return df

def solve_joint(data, npartitions=12,theta=None):
    # load data
    df = data.loc[:,:]
    df[['buy_'+ p + '_joint' for p in ['ann','ltci','rmr']]] = np.nan
    df[['buy_'+ p + '_indp' for p in ['ann','ltci','rmr']]] = np.nan
    df['seed'] = np.random.shuffle(df.index.to_numpy())
    list_df = np.array_split(df, npartitions)
    compress_compute_chunks = partial(compute_joint_chunks,theta=theta)
    with mp.Pool(processes=npartitions) as pool:
        res = pool.map(compress_compute_chunks, list_df)
    df = pd.concat(res)
    return df

def simulate_df(data, npartitions=12,theta=None):
    # load data
    df = data.loc[:,:]
    df[['cons_'+str(x) for x in range(45)]] = np.nan
    df[['own_'+str(x) for x in range(45)]] = np.nan
    df[['wlth_'+str(x) for x in range(45)]] = np.nan
    df[['home_'+str(x) for x in range(45)]] = np.nan
    df['seed'] = np.random.shuffle(df.index.to_numpy())
    list_df = np.array_split(df, npartitions)
    compress_compute_chunks = partial(compute_sim_chunks,theta=theta)
    with mp.Pool(processes=npartitions) as pool:
        res = pool.map(compress_compute_chunks, list_df)
    df = pd.concat(res)
    return df




