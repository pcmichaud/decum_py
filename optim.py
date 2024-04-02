import pandas as pd
from frame import *
import statsmodels.api as sm

def within_difference(data):
        mean_data = data.mean(axis=1)
        for c in data.columns:
                data.loc[:,c] = data[c] - mean_data.loc[:]
        return data

def set_theta(pars, isfree):
        theta = np.zeros(int(np.sum(isfree)))
        i = 0
        # gamma
        if isfree[0]==1:
                theta[i] = pars[0]
                i +=1
        # sigma
        if isfree[1]==1:
                theta[i] = pars[1]
                i +=1
        # rho
        if isfree[2]==1:
                theta[i] = -np.log((1-pars[2])/pars[2])
                i +=1
        # b_x
        if isfree[3]==1:
                theta[i] = pars[3]
                i +=1
        # b_k
        if isfree[4]==1:
                theta[i] = pars[4]
                i +=1
        # nu_c1
        if isfree[5]==1:
                theta[i] = np.log(pars[5])
                i +=1
        # nu_c2
        if isfree[6]==1:
                theta[i] = np.log(pars[6])
                i +=1
        # nu_h
        if isfree[7]==1:
                theta[i] = np.log(pars[7])
                i +=1
        return theta

def extract_pars(theta, isfree, ipars):
        pars = np.zeros(ipars.shape[0])
        i = 0
        # gamma
        if isfree[0]==1:
                pars[0] = theta[i]
                i +=1
        else :
                pars[0] = ipars[0]
        # sigma
        if isfree[1]==1:
                pars[1] = theta[i]
                i +=1
        else :
                pars[1] = ipars[1]
        # rho
        if isfree[2]==1:
                pars[2] = 1.0/(1.0+np.exp(-theta[i]))
                i +=1
        else :
                pars[2] = ipars[2]
        # b_x
        if isfree[3]==1:
                pars[3] = theta[i]
                i +=1
        else :
                pars[3] = ipars[3]
        # b_k
        if isfree[4]==1:
                pars[4] = theta[i]
                i +=1
        else :
                pars[4] = ipars[4]
        # nu_c1
        if isfree[5]==1:
                pars[5] = np.exp(theta[i])
                i +=1
        else :
                pars[5] = ipars[5]
        # nu_c2
        if isfree[6]==1:
                pars[6] = np.exp(theta[i])
                i +=1
        else :
                pars[6] = ipars[6]
        # nu_h
        if isfree[7]==1:
                pars[7] = np.exp(theta[i])
                i +=1
        else :
                pars[7] = ipars[7]
        return pars


def concentrated_distance_within(theta, grad, data, isfree, ipars, iwithin, scn_name, iann = True, irmr = True, iltc = True, npartitions=50):
        # get params
        pars = extract_pars(theta,isfree,ipars)
        # get dataset with solved expected utilities
        df = solve_df(data, iann=iann, iltc = iltc, irmr = irmr, npartitions=npartitions, theta=pars)
        # take difference in value with respect to baseline
        scns = [s for s in range(1,13)]
        for s in scns:
            if s<=4:
                if iann:
                    df['d_value_'+str(s)] = df['value_'+str(s)] - df['value_0']
            if s>=5 and s<=8:
                if iltc:
                     df['d_value_'+str(s)] = df['value_'+str(s)] - df['value_0']
            if s>=9:
                if irmr:
                     df['d_value_'+str(s)] = df['value_'+str(s)] - df['value_0']
        # for each set of products, take within differences
        if iwithin:
            if iann:
                df[['w_value_'+str(s) for s in range(1,5)]] = within_difference(df[['d_value_'+str(s) for s in range(1,5)]])
            if iltc:
                df[['w_value_'+str(s) for s in range(5,9)]] = within_difference(df[['d_value_'+str(s) for s in range(5,9)]])
            if irmr:
                df[['w_value_'+str(s) for s in range(9,13)]] = within_difference(df[['d_value_'+str(s) for s in range(9,13)]])
        else :
            if iann:
                df[['w_value_'+str(s) for s in range(1,5)]] = df[['d_value_'+str(s) for s in range(1,5)]].copy()
            if iltc:
                df[['w_value_'+str(s) for s in range(5,9)]] = df[['d_value_'+str(s) for s in range(5,9)]].copy()
            if irmr:
                df[['w_value_'+str(s) for s in range(9,13)]] = df[['d_value_'+str(s) for s in range(9,13)]].copy()

        # take exp odds transform of probabilities
        s = 1
        for i in range(1,5):
                if iann:
                    df['odd_'+str(s)] = np.log(df['prob_scn_ann_'+str(i)])/(1.0 - df['prob_scn_ann_'+str(i)])
                s += 1
        for i in range(1,5):
                if iltc:
                    df['odd_'+str(s)] = np.log(df['prob_scn_ltci_'+str(i)])/(1.0 - df['prob_scn_ltci_'+str(i)])
                s += 1
        for i in range(1,5):
                if irmr:
                    df['odd_'+str(s)] = np.log(df['prob_scn_rmr_'+str(i)])/(1.0 - df['prob_scn_rmr_'+str(i)])
                s += 1
        # take within deviations
        if iwithin:
            if iann:
                df[['w_odd_'+str(s) for s in range(1,5)]] = within_difference(df[['odd_'
                                                        +str(s) for s in range(1,5)]])
            if iltc:
                df[['w_odd_'+str(s) for s in range(5,9)]] = within_difference(df[['odd_'
                                                        +str(s) for s in range(5,9)]])
            if irmr:
                df[['w_odd_'+str(s) for s in range(9,13)]] = within_difference(df[['odd_'
                                                        +str(s) for s in range(9,13)]])
        else :
            if iann:
                df[['w_odd_'+str(s) for s in range(1,5)]] = df[['odd_'
                                                        +str(s) for s in range(1,5)]].copy()
            if iltc:
                df[['w_odd_'+str(s) for s in range(5,9)]] = df[['odd_'
                                                        +str(s) for s in range(5,9)]].copy()
            if irmr:
                df[['w_odd_'+str(s) for s in range(9,13)]] = df[['odd_'
                                                        +str(s) for s in range(9,13)]].copy()
        # perform OLS to obtain estimates of sigma per product
        sigmas = np.zeros((3,2))
        sum_distance = 0.0
        if iann:
            for k in [0,1]:
                cond = (~df['w_odd_1'].isna()) & (df['know_ann']==k)
                y = df.loc[cond,['w_odd_'+str(s) for s in range(1,5)]].stack().values
                X = df.loc[cond,['w_value_'+str(s) for s in range(1,5)]].stack().values
                X = sm.add_constant(X)
                model = sm.OLS(y,X,missing='drop')
                results = model.fit()
                sigmas[0,k] = results.params[1]
                sum_distance += results.ssr
        if iltc:
            for k in [0,1]:
                cond = (~df['w_odd_5'].isna()) & (df['know_ltci']==k)
                y = df.loc[cond,['w_odd_'+str(s) for s in range(5,9)]].stack().values
                X = df.loc[cond,['w_value_'+str(s) for s in range(5,9)]].stack().values
                X = sm.add_constant(X)
                model = sm.OLS(y,X,missing='drop')
                results = model.fit()
                sigmas[1,k] = results.params[1]
                sum_distance += results.ssr
        if irmr:
            for k in [0,1]:
                cond = (~df['w_odd_9'].isna()) & (df['know_rmr']==k)
                y = df.loc[cond,['w_odd_'+str(s) for s in range(9,12)]].stack().values
                X = df.loc[cond,['w_value_'+str(s) for s in range(9,12)]].stack().values
                X = sm.add_constant(X)
                model = sm.OLS(y,X,missing='drop')
                results = model.fit()
                sigmas[2,k] = results.params[1]
                sum_distance += results.ssr
        print('- function call summary')
        print('ssd = ', sum_distance, ' sigmas = ',sigmas)
        print('pars = ',pars)
        if iann:
            print('std.dev of utility differences - annuities',df[['d_value_'+str(s) for s in range(1,5)]].std())
        if iltc:
            print('std.dev of utility differences - ltc',df[['d_value_'+str(s) for s in range(5,9)]].std())
        if irmr:
            print('std.dev of utility differences - rmr',df[['d_value_'+str(s) for s in range(9,13)]].std())
        np.save('output/sigmas_'+scn_name,sigmas)
        return sum_distance

def residuals_within(theta, sigmas, data, isfree, ipars, npartitions=50):
        # get params
        pars = extract_pars(theta,isfree,ipars)
        # get dataset with solved expected utilities
        df = solve_df(data, npartitions=npartitions, theta=pars)
        # take difference in value with respect to baseline
        scns = [s for s in range(1,13)]
        for s in scns:
                df['d_value_'+str(s)] = df['value_'+str(s)] - df['value_0']
        # for each set of products, take within differences
        df[['w_value_'+str(s) for s in range(1,5)]] = within_difference(df[['d_value_'
                                                        +str(s) for s in range(1,5)]])
        df[['w_value_'+str(s) for s in range(5,9)]] = within_difference(df[['d_value_'
                                                        +str(s) for s in range(5,9)]])
        df[['w_value_'+str(s) for s in range(9,13)]] = within_difference(df[['d_value_'
                                                        +str(s) for s in range(9,13)]])
        # take exp odds transform of probabilities
        s = 1
        for i in range(1,5):
                df['odd_'+str(s)] = np.log(df['prob_scn_ann_'+str(i)])/(1.0 - df['prob_scn_ann_'+str(i)])
                s += 1
        for i in range(1,5):
                df['odd_'+str(s)] = np.log(df['prob_scn_ltci_'+str(i)])/(1.0 - df['prob_scn_ltci_'+str(i)])
                s += 1
        for i in range(1,5):
                df['odd_'+str(s)] = np.log(df['prob_scn_rmr_'+str(i)])/(1.0 - df['prob_scn_rmr_'+str(i)])
                s += 1
        # take within deviations
        df[['w_odd_'+str(s) for s in range(1,5)]] = within_difference(df[['odd_'
                                                        +str(s) for s in range(1,5)]])
        df[['w_odd_'+str(s) for s in range(5,9)]] = within_difference(df[['odd_'
                                                        +str(s) for s in range(5,9)]])
        df[['w_odd_'+str(s) for s in range(9,13)]] = within_difference(df[['odd_'
                                                        +str(s) for s in range(9,13)]])
        residuals = pd.DataFrame(index=df.index,columns=[x for x in range(1,13)])
        df['sigma_ann'] = np.where(df['know_ann']==1,sigmas[0,1],sigmas[0,0])
        df['sigma_ltc'] = np.where(df['know_ltci']==1,sigmas[1,1],sigmas[1,0])
        df['sigma_rmr'] = np.where(df['know_rmr']==1,sigmas[2,1],sigmas[2,0])
        for s in range(1,5):
                residuals.loc[:,s] = df.loc[:,'w_odd_'+str(s)] - df['sigma_ann']* df.loc[:,'w_value_'+str(s)]
        for s in range(5,9):
                residuals.loc[:,s] = df.loc[:,'w_odd_'+str(s)] - df['sigma_ltc']* df.loc[:,'w_value_'+str(s)]
        for s in range(9,13):
                residuals.loc[:,s] = df.loc[:,'w_odd_'+str(s)] - df['sigma_rmr']* df.loc[:,'w_value_'+str(s)]
        return residuals

def g_within(theta, sigmas, data, isfree, ipars, npartitions=50):
        # get params
        pars = extract_pars(theta,isfree,ipars)
        # get dataset with solved expected utilities
        df = solve_df(data, npartitions=npartitions, theta=pars)
        # take difference in value with respect to baseline
        scns = [s for s in range(1,13)]
        for s in scns:
                df['d_value_'+str(s)] = df['value_'+str(s)] - df['value_0']
        # for each set of products, take within differences
        df[['w_value_'+str(s) for s in range(1,5)]] = within_difference(df[['d_value_'
                                                        +str(s) for s in range(1,5)]])
        df[['w_value_'+str(s) for s in range(5,9)]] = within_difference(df[['d_value_'
                                                        +str(s) for s in range(5,9)]])
        df[['w_value_'+str(s) for s in range(9,13)]] = within_difference(df[['d_value_'
                                                        +str(s) for s in range(9,13)]])
        gs = pd.DataFrame(index=df.index,columns=[x for x in range(1,13)])
        df['sigma_ann'] = np.where(df['know_ann']==1,sigmas[0,1],sigmas[0,0])
        df['sigma_ltc'] = np.where(df['know_ltci']==1,sigmas[1,1],sigmas[1,0])
        df['sigma_rmr'] = np.where(df['know_rmr']==1,sigmas[2,1],sigmas[2,0])
        for s in range(1,5):
                gs.loc[:,s] = df['sigma_ann'] * df.loc[:,'w_value_'+str(s)]
        for s in range(5,9):
                gs.loc[:,s] = df['sigma_ltc']* df.loc[:,'w_value_'+str(s)]
        for s in range(9,13):
                gs.loc[:,s] = df['sigma_rmr'] * df.loc[:,'w_value_'+str(s)]
        return gs
