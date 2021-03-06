from solve import *
import multiprocessing as mp
from functools import partial

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




