import numpy as np

def parse_surv(pars):
    gammas = np.zeros(4, dtype=np.float64)
    deltas = np.zeros((3, 4), dtype=np.float64)
    gammas[1] = pars['gamma(2,1)']
    gammas[2] = pars['gamma(3,1)']
    gammas[3] = pars['gamma(4,1)']
    deltas[0, 1] = pars['delta(1,2)']
    deltas[1, 1] = pars['delta(2,2)']
    deltas[2, 1] = pars['delta(3,2)']
    deltas[0, 2] = pars['delta(1,3)']
    deltas[1, 2] = pars['delta(2,3)']
    deltas[2, 2] = pars['delta(3,3)']
    deltas[0, 3] = pars['delta(1,4)']
    deltas[1, 3] = pars['delta(2,4)']
    deltas[2, 3] = pars['delta(3,4)']
    return gammas, deltas

def transition_rates(base_age, gammas, deltas, xi,  miss, miss_par, T, h = 0.0):
    qs = np.zeros((4, 4, T), dtype=np.float64)
    # apply multinomial logit formula
    for i in range(T):
        for j in range(3):
            for k in range(4):
                if k < 3:
                    qs[j, k, i] = np.exp(gammas[k] * float(base_age + i - 60) +
                                       deltas[j, k])
                else :
                    if miss == 0:
                        qs[j, k, i] = np.exp(gammas[k] * float(base_age +
                                        i - 60) + deltas[j, k] + xi)
                    if miss == 1:
                        qs[j, k, i] = np.exp(gammas[k] * float(base_age + i -
                                    60) + deltas[j, k] + miss_par)
                    if (base_age + i) >= 85:
                        qs[j, k, i] *= np.exp(h*gammas[k]*float(base_age + i-85))
            denom = np.sum(qs[j, :, i])
            qs[j,:,i] = qs[j, :, i] / denom
        qs[3, 3, i] = 1.0
        p2 = qs[:, :, i]
        w,v = np.linalg.eig(p2)
        sw = np.diag(np.sqrt(w))
        p1 = np.dot(np.dot(v,sw), np.linalg.inv(v))
        qs[:, :, i] = p1
    return qs

def life_exp(qs,hlth,T):
    sx = np.zeros(T, dtype=np.float64)
    sx[0] = 1
    prob = np.zeros((4, 1))
    prob[hlth] = 1.0
    for i in range(T):
        prob = np.dot(qs[:,:,i].transpose(), prob)
        sx[i] = 1-prob[3, 0]
    return np.sum(sx)

def joint_surv_rates(q1, q1_sp, n_s, T):
    qs_ij = np.zeros((n_s, n_s, T),
                     dtype=np.float64)
    for i in range(T):
        qs_ij[:,:,i] = np.kron(q1[:,:,i],q1_sp[:,:,i])
    return qs_ij




