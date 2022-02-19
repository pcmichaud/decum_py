from solve import *
import multiprocessing as mp


def func(row):
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

    p_h, f_h, p_r, y_ij, med_ij, qs_ij, dims, rates = \
        setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp,
                      surv_bias,sp_surv_bias)
    prefs = set_prefs(live_fast=row['pref_live_fast'],risk_averse=row[
        'pref_risk_averse'],beq_money=row['pref_beq_money'],pref_home=row[
        'pref_home'])

    nu_ij_c,nu_ij_h = update_nus(hh['married'], dims.s_i, dims.s_j, dims, prefs)
    for i in range(13):
        f_prices, f_benfs = set_scenario(row,i)
        i_prices = set_prices(f_prices[0], f_prices[1], f_prices[2], dims.n)
        i_benfs = set_benfs(f_benfs[0], f_benfs[1], f_benfs[2], dims.n)
        b_its = reimburse_loan(i_benfs, i_prices, p_h, dims, rates)
        row['value_' + str(i)] = get_value(hh, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                         p_r, y_ij, med_ij, qs_ij, b_its,
                                         nu_ij_c, nu_ij_h, rates, dims, prefs)
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


def compute_chunks(df):
    df = df.apply(func,axis=1)
    return df

def solve_df(npartitions=4):
    # load data
    data = init_data()
    data[['value_'+str(x) for x in range(13)]] = np.nan
    data = data.iloc[:24,:]
    #data = data.iloc[0,:]
    #data = func(data)
    list_df = np.array_split(data, npartitions)
    with mp.Pool(processes=npartitions) as pool:
        res = pool.map(compute_chunks, list_df)
    data = pd.concat(res)

    return data






