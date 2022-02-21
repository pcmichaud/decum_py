from budget import *

def init_data():
    # load data components
    df_hh = load_hh()
    df_rp = load_rp()
    df_sp = load_sp()
    df_shifters = load_shifters()
    df_know = load_know()
    df_hp, df_hp_sp, df_survival_bias = load_survival()
    df_house_bias = load_house_expectations()
    # parameters of the scenarios
    df_prices, df_benfs, df_probs = load_scenarios()
    # merge datasets
    df = df_hh.merge(df_rp,left_index=True,right_index=True,how='left')
    df = df.merge(df_sp,left_index=True,right_index=True,how='left')
    df = df.merge(df_shifters,left_index=True,right_index=True,how='left')
    df = df.merge(df_know,left_index=True,right_index=True,how='left')
    df = df.merge(df_hp,left_index=True,right_index=True,how='left')
    df_hp_sp.columns = ['sp_'+x for x in df_hp.columns]
    df = df.merge(df_hp_sp,left_index=True,right_index=True,how='left')
    df = df.merge(df_survival_bias,left_index=True,right_index=True,how='left')
    df = df.merge(df_house_bias,left_index=True,right_index=True,how='left')
    df = df.merge(df_prices,left_index=True,right_index=True,how='left')
    df = df.merge(df_benfs,left_index=True,right_index=True,how='left')
    df = df.merge(df_probs,left_index=True,right_index=True,how='left')
    # assign costs
    df_nh, df_hc = load_costs()
    df_house_prices = load_house_prices()
    df[['nh_0','nh_1','nh_2']] = 0.0
    df[['hc_0','hc_1','hc_2']] = 0.0
    df['g'] = 0.0
    df['sig'] = 0.0
    df['base_value'] = 0.0
    for i in df.index:
        df.loc[i,['nh_0','nh_1','nh_2']] = df_nh.loc[df.loc[i,'cma'],[0,1,
                                                                      2]].values
        df.loc[i, ['hc_0', 'hc_1', 'hc_2']] = df_hc.loc[df.loc[i, 'cma'], [0,
                                                                          1,
                                                                           2]].values
        df.loc[i,'g'] = df_house_prices.loc[df.loc[i,'cma'],'g']
        df.loc[i,'sig'] = df_house_prices.loc[df.loc[i,'cma'],'sig']
        df.loc[i,'base_value'] = df_house_prices.loc[df.loc[i,'cma'],
                                                     'base_value']
    return df



def load_hh(file='hh.csv'):
    dtypes = {'respid':'Int64','wgt':'float64','cma':'Int64',
              'married':'Int64','own':'Int64','wealth_total':'float64',
              'mort_balance':'float64','home_value':'float64'}
    df = pd.read_csv('inputs/'+file,dtype=dtypes)
    df.set_index('respid',inplace=True)
    df['wealth_total'] *= 1e-3
    df['mort_balance'] *= 1e-3
    df['home_value'] *= 1e-3
    return df

def load_rp(file='rp.csv'):
    dtypes = {'respid':'Int64','age':'Int64','totinc':'float64',
              'retinc':'float64','hlth':'Int64'}
    df = pd.read_csv('inputs/'+file,dtype=dtypes)
    df.set_index('respid',inplace=True)
    df['totinc'] *= 1e-3
    df['retinc'] *= 1e-3
    return df

def load_sp(file='sp.csv'):
    dtypes = {'respid':'Int64','sp_age':'Int64','sp_totinc':'float64',
              'sp_retinc':'float64','sp_hlth':'Int64'}
    df = pd.read_csv('inputs/'+file,dtype=dtypes)
    df.set_index('respid',inplace=True)
    df = df[df.sp_age!=0]
    df['sp_totinc'] *= 1e-3
    df['sp_retinc'] *= 1e-3
    return df

def load_shifters(file='pf_rp.csv'):
    dtypes = {'respid':'Int64','pref_beq_money':'Int64',
              'pref_home':'Int64','pref_live_fast':'Int64','pref_risk_averse':'Int64'}
    df = pd.read_csv('inputs/'+file,dtype=dtypes)
    df.set_index('respid',inplace=True)
    return df
def load_survival(file='survival_expectations.csv'):
    dtypes = {'respid':'Int64','xi':'float64',
              'xi_sp':'float64','miss_psurv85':'Int64',
              'sp_miss_psurv85':'Int64'}
    df = pd.read_csv('inputs/'+file,dtype=dtypes)
    df.set_index('respid',inplace=True)
    for c in ['xi','xi_sp']:
        df[c] = np.where(df[c]==-9.0,np.nan,df[c])
    for c in ['miss_psurv85','sp_miss_psurv85']:
        df[c] = np.where(df[c]==-9.0,np.nan,df[c])
    df_hp = load_hp()
    df_hp_sp = load_hp_sp()
    return df_hp, df_hp_sp, df
def load_house_expectations(file='house_expectations.csv'):
    df = pd.read_csv('inputs/'+file)
    df.set_index('respid',inplace=True)
    return df
def load_scenarios(price_file='prices.csv',benfs_file='benefits.csv', 
        probs_file='prob.csv'):
    df_prices = pd.read_csv('inputs/'+price_file)
    df_prices.set_index('respid',inplace=True)
    for c in df_prices.columns:
        if 'ann' in c or 'ltci' in c:
            df_prices[c] = np.where(df_prices[c] == -999, 0, df_prices[c])
            df_prices[c] *= 1e-3
    df_benfs = pd.read_csv('inputs/'+benfs_file)
    df_benfs.set_index('respid',inplace=True)
    for c in df_benfs.columns:
            df_benfs[c] = np.where(df_benfs[c]==-999, 0, df_benfs[c])
            df_benfs[c] *= 1e-3
    df_probs = pd.read_csv('inputs/'+probs_file)
    df_probs.set_index('respid',inplace=True)
    for c in df_probs.columns:
        df_probs[c] = np.where(df_probs[c]==-999,np.nan,df_probs[c])
        df_probs[c] = np.where(df_probs[c]==0,0.01,df_probs[c])
        df_probs[c] = np.where(df_probs[c]==1,0.99,df_probs[c])
    return df_prices, df_benfs, df_probs

def load_know(file='know.csv'):
    df = pd.read_csv('inputs/'+file,dtype='Int64')
    df.set_index('respid',inplace=True)
    return df
def load_hp(file='hp.csv'):
    df = pd.read_csv('inputs/'+file)
    df = df.rename({'id':'respid'},axis=1)
    df.set_index('respid',inplace=True)
    return df

def load_hp_sp(file='hp_sp.csv'):
    df = pd.read_csv('inputs/'+file)
    df.set_index('respid',inplace=True)
    df = df[df['gamma(2,1)']!=-9.0]
    return df




