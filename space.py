from numba import njit, float64, int64
from numba.experimental import jitclass
from budget import *

spec_dims = [
    ('T',int64),
    ('t_last',int64),
    ('nper',int64),
    ('time_t',int64[:]),
    ('e_min',float64),
    ('e_max',float64),
    ('n_e',int64),
    ('e_space',float64[:]),
    ('d_min',float64),
    ('d_max',float64),
    ('n_d',int64),
    ('d_space',float64[:]),
    ('d_h',float64[:,:,:]),
    ('n_s',int64),
    ('s_ij',int64[:]),
    ('s_i',int64[:]),
    ('s_j',int64[:]),
    ('ij_h',int64[:]),
    ('a_i',int64[:]),
    ('a_j',int64[:]),
    ('w_min',float64),
    ('w_max',float64),
    ('n_w',int64),
    ('n_h',int64),
    ('w_space',float64[:,:,:,:,:,:]),
    ('n_state_vars',int64),
    ('n_states',int64),
    ('grid_states',int64[:,:]),
    ('is_adm',int64[:]),
    ('n_adm',int64),
    ('adm',int64[:,:]),
    ('to_states',int64[:])
]
@jitclass(spec_dims)
class set_dims(object):
    def __init__(self, married, omega_d, T = 35, e_min = -2.0, e_max = 2.0,
                    n_e = 5, d_min = 0.0, n_d = 5, w_min = 0.0, w_max = 2.0e3,
                    n_w = 10, n_h = 2):
        self.T = T
        self.t_last = 33
        #self.time_t = np.arange(self.t_last, dtype=np.int64)
        self.time_t = np.array([0,1,5,10,20,30,32,33])
        self.nper = self.time_t.shape[0]
        self.e_min = e_min
        self.e_max = e_max
        self.n_e = n_e
        self.e_space = np.linspace(e_min, e_max, n_e)
        self.d_min = d_min
        self.d_max = omega_d
        self.n_d = n_d
        self.d_space = np.linspace(self.d_min, self.d_max, self.n_d)
        self.d_h = np.empty((self.n_d, self.n_e, self.T))

        if married==1:
            self.n_s = 16
            self.s_ij = np.arange(self.n_s,dtype=np.int64)
            self.s_i = np.empty(self.n_s,dtype=np.int64)
            self.s_j = np.empty(self.n_s,dtype=np.int64)
            for i in range(4):
                for j in range(4):
                    self.s_i[i*4 + j] = i
                    self.s_j[i*4 + j] = j
            self.ij_h = np.empty(self.n_s,dtype=np.int64)
            for i in range(self.n_s):
                if (min(self.s_i[i], self.s_j[i]) <= 1):
                    self.ij_h[i] = 1
                else :
                    self.ij_h[i] = 0
        else :
            self.n_s = 4
            self.s_ij = np.arange(self.n_s,dtype=np.int64)
            self.s_i = self.s_ij[:]
            self.s_j = self.s_ij[:]
            self.ij_h = np.where(self.s_i <= 1, 1, 0)
        self.a_i = np.where(self.s_i < 3, 1, 0)
        self.a_j = np.where(self.s_j < 3, 1, 0)
        self.w_min = w_min
        self.w_max = w_max
        self.n_w = n_w
        self.n_h = n_h
        self.w_space = np.empty((self.n_d, self.n_w, self.n_s, self.n_e,
                                 self.n_h, self.T))
        self.n_state_vars = 5
        self.n_states = self.n_h * self.n_e * self.n_s * self.n_w * \
                        self.n_d
        self.grid_states = np.empty((self.n_states, self.n_state_vars),
                                    dtype=np.int64)
        self.is_adm = np.empty(self.n_states,dtype=np.int64)
        i = 0
        for h in range(self.n_h):
            for e in range(self.n_e):
                for s in range(self.n_s):
                    for w in range(self.n_w):
                        for d in range(self.n_d):
                            self.grid_states[i, :] = np.array([d, w, s, e, h])
                            self.is_adm[i] = 1
                            if s == self.n_s - 1:
                                self.is_adm[i] = 0
                            if h == 0 and d > 0:
                                self.is_adm[i] = 0
                            i += 1
        self.n_adm = np.sum(self.is_adm)
        self.adm = np.empty((self.n_adm, self.n_state_vars),dtype=np.int64)
        self.to_states = np.empty(self.n_adm,dtype=np.int64)
        j = 0
        for i in range(self.n_states):
            if self.is_adm[i] == 1:
                self.adm[j, :] = self.grid_states[i, :]
                self.to_states[j] = i
                j += 1
        return
    def set_wspace(self,y_ij, p_h, rates):
        for j in range(self.T):
                y = y_ij[:,j]
                for h in range(self.n_h):
                    for e in range(self.n_e):
                        for d in range(self.n_d):
                            for s in range(self.n_s):
                                if h==0:
                                    x_w = -rates.omega_r * y[s]
                                else :
                                    d0 = self.d_h[d, e, j]
                                    a = rates.omega_h0 * p_h[e, j]
                                    b = rates.omega_h1 * max(p_h[e, j] - d0,
                                                             0.0)
                                    c = rates.omega_r * y[s]
                                    x_w = -min(min(a, b),c)
                                self.w_space[d,:,s,e,h,j] = \
                                                np.linspace(x_w,self.w_max,
                                                                       self.n_w)
        return
    def set_dspace(self, p_h):
        for j in range(self.T):
            for k in range(self.n_e):
                for i in range(self.n_d):
                    self.d_h[i,k,j] = self.d_space[i] * p_h[k,j]
        return

