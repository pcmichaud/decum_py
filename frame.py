from solve import *
import multiprocessing as mp
from functools import partial
from scipy.interpolate import interp1d
from itertools import product

def get_actors(row):
    # household information
    hh = dict(row[['wgt','cma','married','own','wealth_total',
                   'mort_balance','home_value']])
    rp = dict(row[['age','totinc','retinc','hlth']])
    if hh['married'] == 1:
        sp = dict(row[['sp_age','sp_totinc','sp_retinc','sp_hlth']])
    else:
        sp = None
    return hh, rp, sp

def get_house_info(row,debias=False,g_fudge=1.0,sig_fudge=1.0):
    if debias:
        g = row['mu']
        sig = row['sig']
    else :
        g = row['g'] * row['mu']
        sig = row['sig'] * row['zeta']
    g *= g_fudge
    sig *= sig_fudge
    base_value = row['base_value']
    return g, sig, base_value

def get_medcosts(row, fudge_hc = 1.0, fudge_nh = 1.0):
    hc = row[['hc_0','hc_1','hc_2']].to_numpy(dtype='float64')
    nh = row[['nh_0','nh_1','nh_2']].to_numpy(dtype='float64')
    hc *= fudge_hc
    nh *= fudge_nh
    return hc, nh

def get_health_params(row):
    hp_vars = ['gamma(2,1)', 'delta(1,2)', 'delta(2,2)', 'delta(3,2)',
               'gamma(3,1)', 'delta(1,3)', 'delta(2,3)', 'delta(3,3)',
               'gamma(4,1)', 'delta(1,4)', 'delta(2,4)', 'delta(3,4)']
    hp = row[hp_vars]
    surv_bias = row[['xi','miss_psurv85']]
    if row['married'] == 1:
        hp_sp_vars = ['sp_' + x for x in hp_vars]
        hp_sp = row[hp_sp_vars]
        hp_sp.index = hp_vars
        sp_surv_bias = row[['xi_sp','sp_miss_psurv85']]
    else:
        hp_sp = None
        sp_surv_bias = None
    return hp, surv_bias, hp_sp, sp_surv_bias

def get_prefs(row,theta):
    prefs = set_prefs(gamma=theta[0],vareps=theta[1],rho=theta[2],
                        b_x=theta[3],b_k=theta[4],nu_c1=theta[5],
                        nu_c2=theta[6],nu_h=theta[7])
    return prefs

# function used to solve for expected value across scenarios
def func_solve(row,theta, iann, iltc, irmr):
    # household information
    hh, rp, sp = get_actors(row)

    # house price information
    g, sig, base_value = get_house_info(row)

    # health transition parameters
    hc, nh = get_medcosts(row)

    # health transition parameters
    hp, surv_bias, hp_sp, sp_surv_bias = get_health_params(row)

    # setup the problem, depending on whether preferences supplied or not
    prefs = get_prefs(row,theta)
    p_h, f_h, p_r, y_ij, med_ij, qs_ij, dims, rates = \
        setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp,
                    surv_bias,sp_surv_bias,miss_par=0.0,sp_miss_par=0.0)
    nu_ij_c = update_nus(hh['married'], dims.s_i, dims.s_j, dims, prefs, rates.eqscale)

    # run the task for this function: get value for all 13 scenarios (12 + baseline)
    v_t = np.zeros((dims.n_states,dims.T),dtype='float64')
    v_ref = np.zeros((dims.n_states,dims.T),dtype='float64')
    for i in range(13):
        hhp = hh.copy()
        f_prices, f_benfs = set_scenario(row,i)
        i_prices = set_prices(f_prices[0], f_prices[1], f_prices[2])
        i_benfs = set_benfs(f_benfs[0], f_benfs[1], f_benfs[2])
        if i_benfs.rmr>0.0:
            if hhp['mort_balance']>0:
                if i_benfs.rmr>hhp['mort_balance']:
                    i_benfs.rmr -= hhp['mort_balance']
                    hhp['mort_balance'] = 0
                else :
                    i_benfs.rmr = 0.0
        b_its = reimburse_loan(i_benfs, i_prices, p_h, dims, rates)
        if i==0:
            row['value_' + str(i)], v_ref = get_value(hhp, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                         p_r, y_ij, med_ij, qs_ij, b_its,
                                         nu_ij_c, rates, dims, prefs, v_t)
        if i>=1 and i<=4:
            if iann:
                row['value_' + str(i)], v_t  = get_value(hhp, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                         p_r, y_ij, med_ij, qs_ij, b_its,
                                         nu_ij_c, rates, dims, prefs, v_ref)
        if i>=5 and i<=8:
             if iltc:
                row['value_' + str(i)], v_t = get_value(hhp, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                         p_r, y_ij, med_ij, qs_ij, b_its,
                                         nu_ij_c, rates, dims, prefs, v_ref)
        if i>=9 & i<=12:
            if irmr:
                row['value_' + str(i)], v_t = get_value(hhp, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                         p_r, y_ij, med_ij, qs_ij, b_its,
                                         nu_ij_c, rates, dims, prefs, v_ref)
    return row


def func_sim(row,theta):
    # household information
    hh, rp, sp = get_actors(row)

    # house price information
    g, sig, base_value = get_house_info(row)

    # health transition parameters
    hc, nh = get_medcosts(row)

    # health transition parameters
    hp, surv_bias, hp_sp, sp_surv_bias = get_health_params(row)

    # setup the problem, depending on whether preferences supplied or not
    prefs = get_prefs(row,theta)
    p_h, f_h, p_r, y_ij, med_ij, qs_ij, dims, rates = \
        setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp,
                    surv_bias,sp_surv_bias,miss_par=0.0,sp_miss_par=0.0)
    nu_ij_c = update_nus(hh['married'], dims.s_i, dims.s_j, dims, prefs, rates.eqscale)


    i_prices = set_prices(0.0,0.0 , rates.r_h)
    i_benfs = set_benfs(0.0, 0.0, 0.0)
    b_its = reimburse_loan(i_benfs, i_prices, p_h, dims, rates)
    v_ref = np.zeros((dims.n_states,dims.T),dtype=np.float64)
    cons_rules, cond_values,v_ref = get_rules(hh, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                            p_r, y_ij, med_ij, qs_ij, b_its,
                                            nu_ij_c,rates, dims, prefs, v_ref)
    # record how often wealth is low
    nsim = 1000
    pnone = 0.0
    target = 85 - rp['age']
    for i in range(nsim):
        cons_path, own_path, wlth_path, home_path     = get_sim_path(i,
                                        cons_rules,cond_values, hh, rp, sp, base_value, i_prices,
                                        i_benfs, p_h, f_h, p_r, y_ij, med_ij, qs_ij, b_its,
                                        nu_ij_c,rates, dims, prefs)
        pnone += np.where(wlth_path[target]<=rates.x_min,1.0,0.0)
    row['pexhaust85_sim'] = pnone/float(nsim)
    return row


def func_fair(row,theta):
    # household information
    hh, rp, sp = get_actors(row)

    # house price information
    g, sig, base_value = get_house_info(row)

    # health transition parameters
    hc, nh = get_medcosts(row)

    # health transition parameters
    hp, surv_bias, hp_sp, sp_surv_bias = get_health_params(row)

    # setup the problem, depending on whether preferences supplied or not
    prefs = get_prefs(row,theta)
    p_h, f_h, p_r, y_ij, med_ij, qs_ij, dims, rates = \
        setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp,
                    surv_bias,sp_surv_bias,miss_par=0.0,sp_miss_par=0.0)
    nu_ij_c = update_nus(hh['married'], dims.s_i, dims.s_j, dims, prefs, rates.eqscale)

    # objective health probabilities
    gammas, deltas = parse_surv(hp)
    q1_o = transition_rates(rp['age'], gammas, deltas, 0.0,
                          surv_bias['miss_psurv85'], 0.0, dims.T)
    if hh['married'] == 1:
        gammas, deltas = parse_surv(hp_sp)
        q1_sp_o = transition_rates(sp['sp_age'], gammas, deltas, 0.0,
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
    p_a = np.sum([np.exp(-rates.rate*i)*sx[i] for i in range(1,dims.T)])
    row['price_ann_fair'] = p_a
    # compute fair price of long-term care insurance (see eq. 3 of paper)
    Lx = np.zeros(dims.T, dtype=np.float64)
    qx = np.zeros(dims.T, dtype=np.float64)
    for i in range(dims.T):
        Lx[i] = np.sum(q_n[dims.s_i==2,i])
        qx[i] = np.sum(q_n[dims.s_i<=1,i])
    num = np.sum([np.exp(-rates.rate*i)*Lx[i] for i in range(1,dims.T)])
    den = np.sum([np.exp(-rates.rate*i)*qx[i] for i in range(dims.T)])
    p_l = num/den
    row['price_ltci_fair'] = p_l
    # find out how much want to buy, set bounds
    opt_max = np.zeros(3,dtype=np.float64)
    opt_min = np.zeros(3,dtype=np.float64)
    opt_max[0] = hh['wealth_total']
    opt_max[1] = med_ij[2]
    opt_max[2] = rates.omega_rm*(hh['home_value'] - hh['mort_balance'])
    row['max_ann_fair'] = opt_max[0]
    row['max_ltci_fair'] = opt_max[1]
    row['max_rmr_fair'] = max(opt_max[2],0.0)

    # find out first values for reference scenario
    i_prices = set_prices(0.0,0.0 , rates.r_h)
    i_benfs = set_benfs(0.0, 0.0, 0.0)
    b_its = reimburse_loan(i_benfs, i_prices, p_h, dims, rates)
    values_ref = np.zeros((dims.n_states,dims.T),dtype=np.float64)
    cons_rules_ref, cond_values_ref, values_ref = get_rules(hh, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                            p_r, y_ij, med_ij, qs_ij, b_its,
                                            nu_ij_c,rates, dims, prefs,values_ref)


    # everything is parametrized to fit into unit cube for demand
    pi = 0.0*np.ones(3,dtype=np.float64)
    pi[2] = 1.0
    i_rs = np.linspace(0.0,0.05,5)
    ds = []
    ps = []
    for i_r in i_rs:
        hhp = hh.copy()
        i_prices = set_prices(pi[0]*opt_max[0],p_l*pi[1]*opt_max[1] , rates.r_h + i_r)
        i_benfs = set_benfs(pi[0]*opt_max[0]/p_a, pi[1]*opt_max[1], pi[2]*opt_max[2])
        if i_benfs.rmr>0.0:
            if hhp['mort_balance']>0:
                if i_benfs.rmr>hhp['mort_balance']:
                    i_benfs.rmr -= hhp['mort_balance']
                    hhp['mort_balance'] = 0
                else :
                    i_benfs.rmr = 0.0
        b_its = reimburse_loan(i_benfs, i_prices, p_h, dims, rates)
        # get decision rules based on fair pricing for annuities and LTCI, candidate rmr
        cons_rules, cond_values, values = get_rules(hhp, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                            p_r, y_ij, med_ij, qs_ij, b_its,
                                            nu_ij_c,rates, dims, prefs, values_ref)
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
            cons_path, own_path, wlth_path, home_path     = get_sim_path(i,
                                        cons_rules, cond_values, hhp, rp, sp, base_value, i_prices,
                                        i_benfs, p_h, f_h, p_r, y_ij, med_ij, qs_ij, b_its,
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


def func_joint(row,theta, ixmin = False, dispose_home = False):
    # household information
    hh, rp, sp = get_actors(row)

    # house price information
    g, sig, base_value = get_house_info(row)

    # health transition parameters
    hc, nh = get_medcosts(row)

    # health transition parameters
    hp, surv_bias, hp_sp, sp_surv_bias = get_health_params(row)

    # setup the problem, depending on whether preferences supplied or not
    prefs = get_prefs(row,theta)
    p_h, f_h, p_r, y_ij, med_ij, qs_ij, dims, rates = \
        setup_problem(hh, rp, sp, g, sig, base_value, hc, nh, hp, hp_sp,
                    surv_bias,sp_surv_bias,miss_par=0.0,sp_miss_par=0.0)
    nu_ij_c = update_nus(hh['married'], dims.s_i, dims.s_j, dims, prefs, rates.eqscale)

    if ixmin:
        rates.x_min = 1.0
    if dispose_home:
        hh['own'] = 0
        hh['wealth_total'] += hh['home_value'] - hh['mort_balance']
        hh['home_value'] = 0.0
        hh['mort_balance'] = 0.0

    # fair prices
    p_a = row['price_ann_fair']
    p_l = row['price_ltci_fair']
    i_r = rates.r_h + row['price_rmr_fair']


    # find out first values for reference scenario
    i_prices = set_prices(0.0,0.0 , rates.r_h)
    i_benfs = set_benfs(0.0, 0.0, 0.0)
    b_its = reimburse_loan(i_benfs, i_prices, p_h, dims, rates)
    values_ref = np.zeros((dims.n_states,dims.T),dtype=np.float64)
    cons_rules_ref, cond_values_ref, values_ref = get_rules(hh, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                            p_r, y_ij, med_ij, qs_ij, b_its,
                                            nu_ij_c,rates, dims, prefs,values_ref)
    # set bounds
    opt_max = np.zeros(3,dtype=np.float64)
    opt_min = np.zeros(3,dtype=np.float64)
    opt_max[0] = hh['wealth_total']
    opt_max[1] = med_ij[2]
    opt_max[2] = max(rates.omega_rm*(hh['home_value'] - hh['mort_balance']),0.0)
    # search over grid
    nu = 2
    nuu = nu*nu*nu
    gridu = np.linspace(0,0.5,nu)
    gridr = np.linspace(0,1.0,nu)
    buy_grid = np.array(list(product(*[gridu,gridu,gridr])))
    values = np.zeros(nuu)
    for i in range(nuu):
        hhp = hh.copy()
        pi = buy_grid[i,:]
        i_prices = set_prices(pi[0]*opt_max[0],p_l*pi[1]*opt_max[1] , i_r)
        i_benfs = set_benfs(pi[0]*opt_max[0]/p_a, pi[1]*opt_max[1], pi[2]*opt_max[2])
        if i_benfs.rmr>0.0:
            if hhp['mort_balance']>0:
                if i_benfs.rmr>hhp['mort_balance']:
                    i_benfs.rmr -= hhp['mort_balance']
                    hhp['mort_balance'] = 0
                else :
                    i_benfs.rmr = 0.0
        b_its = reimburse_loan(i_benfs, i_prices, p_h, dims, rates)
        # get decision rules based on fair pricing for annuities and LTCI, candidate rmr
        values[i], v_t =  get_value(hhp, rp, sp, base_value, i_prices, i_benfs, p_h, f_h,
                                         p_r, y_ij, med_ij, qs_ij, b_its,
                                         nu_ij_c, rates, dims, prefs, values_ref)
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


def compute_solve_chunks(df,theta, iann, iltc, irmr):
    df = df.apply(func_solve,axis=1,args=(theta,iann,iltc,irmr,))
    return df

def compute_fair_chunks(df,theta):
    df = df.apply(func_fair,axis=1,args=(theta,))
    return df

def compute_sim_chunks(df,theta):
    df = df.apply(func_sim,axis=1,args=(theta,))
    return df


def compute_joint_chunks(df,theta, ixmin = False, dispose_home = False):
    df = df.apply(func_joint,axis=1,args=(theta,ixmin,dispose_home))
    return df


def solve_df(data, iann = True, iltc = True, irmr = True, npartitions=12,theta=None):
    # load data
    df = data.loc[:,:]
    df[['value_'+str(x) for x in range(13)]] = np.nan
    list_df = np.array_split(df, npartitions)
    compress_compute_chunks = partial(compute_solve_chunks,theta=theta, iann = iann, iltc = iltc, irmr = irmr)
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

def solve_sim(data, npartitions=12,theta=None):
    # load data
    df = data.loc[:,:]
    df['pexhaust85_sim'] = np.nan
    df['seed'] = np.random.shuffle(df.index.to_numpy())
    list_df = np.array_split(df, npartitions)
    compress_compute_chunks = partial(compute_sim_chunks,theta=theta)
    with mp.Pool(processes=npartitions) as pool:
        res = pool.map(compress_compute_chunks, list_df)
    df = pd.concat(res)
    return df



def solve_joint(data, npartitions=12,theta=None, ixmin = False, dispose_home = False):
    # load data
    df = data.loc[:,:]
    df[['buy_'+ p + '_joint' for p in ['ann','ltci','rmr']]] = np.nan
    df[['buy_'+ p + '_indp' for p in ['ann','ltci','rmr']]] = np.nan
    df['seed'] = np.random.shuffle(df.index.to_numpy())
    list_df = np.array_split(df, npartitions)
    compress_compute_chunks = partial(compute_joint_chunks,theta=theta, ixmin = ixmin, dispose_home = dispose_home)
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




