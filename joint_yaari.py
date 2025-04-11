from frame import *
import warnings
from numba import set_num_threads
import pandas as pd

if __name__ == '__main__':
    warnings.simplefilter(action='ignore')
    data = init_data()
    # load estimated parameters
    theta = np.load('output/estimates_ez.npy')

    data['totinc'] = np.where(data['married']==1,data['totinc']+               data['sp_totinc'], data['totinc'])
    data['retinc'] = np.where(data['married']==1,data['retinc']+               data['sp_retinc'], data['retinc'])
    data['married'] = 0

    #data[['hc_0','hc_1','hc_2']] = 0.0
    #data[['nh_0','nh_1','nh_2']] = 0.0

    theta[3] = 5.0
    theta[4] = 250.0
    theta[5] = 1.0
    #theta[6] = 1.0

    #theta[7] = 0.0
    #data['g'] = 0.0
    #data['sig'] = 0.0

    #data['xi'] = 0.0
    #data['xi_sp'] = 0.0

    # load fair prices
    prices = pd.read_csv('output/fair_prices.csv')
    prices.set_index('respid',inplace=True)
    # load other vars
    others = pd.read_csv('inputs/other_vars.csv')
    others.set_index('respid',inplace=True)

    data = data.merge(prices,left_on='respid',right_on='respid',how='left')
    data = data.merge(others,left_on='respid',right_on='respid',how='left')
    data.loc[data['price_rmr_fair'].isna(),'price_rmr_fair'] = 0.0

    # average prices per cell
    avg_prices = data.groupby(['age','female']).mean()[['price_ann_fair','price_ltci_fair','price_rmr_fair']]
    avg_prices.columns = ['price_ann_fair_avg','price_ltci_fair_avg','price_rmr_fair_avg']
    data = data.merge(avg_prices,left_on=['age','female'],right_on=['age','female'],how='left')

    print(data[['price_ann_fair_avg','price_ltci_fair_avg','price_rmr_fair_avg']].describe().transpose())
    print(data[['price_ann_fair','price_ltci_fair','price_rmr_fair']].describe().transpose())
    values = solve_joint(data, npartitions=250,theta=theta, dispose_home=True)
    for c in ['buy_ann_joint','buy_ltci_joint','buy_rmr_joint']:
        values[c] = values[c].astype('float64')
    for c in ['buy_ann_indp','buy_ltci_indp','buy_rmr_indp']:
        values[c] = values[c].astype('float64')
    print(values[['buy_ann_joint','buy_ltci_joint','buy_rmr_joint']].describe().transpose())
    print(values[['buy_ann_indp','buy_ltci_indp','buy_rmr_indp']].describe().transpose())
    values.to_csv('output/joint_yaari.csv')
