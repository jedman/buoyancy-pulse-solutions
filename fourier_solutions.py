import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'RdBu_r'

class FourierSolution:
    def __init__(self,x,z):
        '''solutions for linearized, hydrostatic, forced boussinesq waves in a two-layer stratified
        atmosphere''' 
        self.x = x
        self.z = z
        self.N1 = 0.01 # base brunt vaisala frequency for troposphere
        self.H = 15000. # tropopause height (m)
        self.tropindex = np.where(z<=1)
        self.stratindex = np.where(z>1)
        self.M = np.pi


    def alpha(self,m,k):
        '''amplitude of forced solution'''
        Qcoeff= m**2/(m**2-self.M**2)/self.N1**2
        #Qcoeff = 1. # COMMENT OUT TO USE PULSE FORCING
        return Qcoeff

    def Bl(self,m,k,eps):
        '''coefficient for rightward reflected solution in the lower layer
        m, M, normalized by tropopause height '''
        w = self.N1*k/m
        Bcoeff = self.alpha(m,k)*\
        (1./(eps**2*np.cos(m)**2 + np.sin(m)**2))*(-eps**2*self.M/m * np.cos(self.M)*np.cos(m)\
                                                 - np.sin(self.M) * np.sin(m)\
                                                 +1j*eps*(-self.M/m*np.cos(self.M)*np.sin(m)\
                                                           + np.cos(m)*np.sin(self.M)))
        return Bcoeff
    def Br(self,m,k,eps):
        '''coefficient forleftward reflected solution in the lower layer
        m, M, normalized by tropopause height '''
        w = self.N1*k/m
        Bcoeff = self.alpha(m,k)*\
        (1./(eps**2*np.cos(m)**2 + np.sin(m)**2))*(-eps**2*self.M/m * np.cos(self.M)*np.cos(m)\
                                                 - np.sin(self.M) *np.sin(m)\
                                                 - 1j*eps*(-self.M/m*np.cos(self.M)*np.sin(m)\
                                                           + np.cos(m)*np.sin(self.M)))
        return Bcoeff

    def Cr(self,m,k,eps):
        #coefficient for reflected solution in the upper layer
        w = self.N1*k/m
        Ccoeff = (self.Br(m,k,eps)*np.sin(m) + self.alpha(m,k)*np.sin(self.M))*np.exp(+1j*m/eps)
        return Ccoeff
    def Cl(self,m,k,eps):
        #coefficient for left  solution in the upper layer
        w = self.N1*k/m
        Ccoeff = (self.Bl(m,k,eps)*np.sin(m) + self.alpha(m,k)*np.sin(self.M))*np.exp(-1j*m/eps)
        return Ccoeff

    def tl(self,w,t, t0 = 0.):
        tdep = np.exp(1j*w*t)
        return tdep
    def tr(self,w,t, t0 = 0.):
        tdep = np.exp(-1j*w*t)
        return tdep

    def reset_data(self):
        '''reset self.data to zeros for new calculation'''
        self.data = np.zeros((len(self.x),len(self.z)), dtype = complex)
        return




    def get_solution(self, t,m, k, eps, strat = True):
        self.reset_data()
        w = self.N1*k/m # omega
        dm = 1.
        dk = 1.

        self.solve_component(t,m,k,eps,dm,dk,solvefor = ('velocity','pressure'), strat='True')

    def solve_component(self,t,m,k,eps,dm=1.,dk=1., solvefor = 'velocity', strat = True):
        '''solution routine for a given m, k
        solvefor (velocity) returns self.state['w','u']
        solvefor (pressure) returns self.state['p'] '''
        # record state input
        self.m = m
        self.k = k
        self.eps = eps
        self.t = t

        self.state = {} ## put all the state variables here

        w = self.N1*k/m # omega
        self.w = w
        if 'velocity' in solvefor:
            self.reset_data()
            self.data[:,self.tropindex[0]] = self.data[:,self.tropindex[0]] \
                + dm*dk*np.outer(np.exp(1j*k*x),(self.alpha(m,k)*(self.tr(w,t)\
                + self.tl(w,t))*np.sin(self.M*z[self.tropindex[0]]) \
                + (self.Bl(m,k,eps)*self.tl(w,t) \
                + self.Br(m,k,eps)*(self.tr(w,t)))*np.sin(m*z[self.tropindex[0]])))

            if (strat):
                self.data[:,self.stratindex[0]] = self.data[:,self.stratindex[0]] \
                    + dm*dk*(np.outer(np.exp(1j*k*x), self.tr(w,t)*self.Cr(m,k,eps) \
                    *np.exp(-1j*m/eps*z[self.stratindex[0]])\
                   + self.tl(w,t)*self.Cl(m,k,eps)\
                    * np.exp(+1j*m/eps*z[self.stratindex[0]])))

            self.state['w'] = self.data

            dx = np.abs(self.x[1]-self.x[0])
            dz = np.abs(self.z[1]-self.z[0])

            # numerical approximation to u, should use analytical solution
            #self.state['u_diff'] = -dz/dx*np.cumsum(np.diff(test.data.T, axis = 0),axis =1)

            # calculate partial_z w
            self.reset_data()
            self.data[:,self.tropindex[0]] = self.data[:,self.tropindex[0]] \
                + dm*dk*np.outer(np.exp(1j*k*x),(self.alpha(m,k)*(self.tr(w,t)\
                + self.tl(w,t))*self.M*np.cos(self.M*z[self.tropindex[0]]) \
                + (self.Bl(m,k,eps)*self.tl(w,t) \
                + self.Br(m,k,eps)*(self.tr(w,t)))*m*np.cos(m*z[self.tropindex[0]])))


            if (strat):
                self.data[:,self.stratindex[0]] = self.data[:,self.stratindex[0]] \
                    + dm*dk*(np.outer(np.exp(1j*k*x), self.tr(w,t)*self.Cr(m,k,eps) \
                    *-1j*m/eps*np.exp(-1j*m/eps*z[self.stratindex[0]])\
                   + self.tl(w,t)*self.Cl(m,k,eps)\
                    * 1j*m/eps*np.exp(+1j*m/eps*z[self.stratindex[0]])))

            self.state['u'] = 1j/k*self.data

        if 'pressure' in solvefor:
            self.reset_data()
            self.data[:,self.tropindex[0]] = self.data[:,self.tropindex[0]] \
                + dm*dk*np.outer(np.exp(1j*k*x),(self.alpha(m,k)*(-1j*w*self.tr(w,t)\
                + 1j*w*self.tl(w,t))*self.M*np.cos(self.M*z[self.tropindex[0]]) \
                + (self.Bl(m,k,eps)*(1j*w)*self.tl(w,t) \
                + self.Br(m,k,eps)*(-1j*w*self.tr(w,t)))*m*np.cos(m*z[self.tropindex[0]])))


            if (strat):
                self.data[:,self.stratindex[0]] = self.data[:,self.stratindex[0]] \
                    + dm*dk*(np.outer(np.exp(1j*k*x), -1j*w*self.tr(w,t)*self.Cr(m,k,eps) \
                   *(-1j*m/eps)*np.exp(-1j*m/eps*z[self.stratindex[0]])\
                   + 1j*w*self.tl(w,t)*self.Cl(m,k,eps)\
                    *1j*m/eps*np.exp(+1j*m/eps*z[self.stratindex[0]])))

            self.state['p'] = -self.data/k**2

            # calculate b = partial_z p
            self.reset_data()
            self.data[:,self.tropindex[0]] = self.data[:,self.tropindex[0]] \
                + dm*dk*np.outer(np.exp(1j*k*x),(self.alpha(m,k)*(-1j*w*self.tr(w,t)\
                + 1j*w*self.tl(w,t))*(-self.M**2)*np.sin(self.M*z[self.tropindex[0]]) \
                + (self.Bl(m,k,eps)*(1j*w)*self.tl(w,t) \
                + self.Br(m,k,eps)*(-1j*w*self.tr(w,t)))*(-m**2)*np.sin(m*z[self.tropindex[0]])))
            if (strat):
                self.data[:,self.stratindex[0]] = self.data[:,self.stratindex[0]] \
                    + dm*dk*(np.outer(np.exp(1j*k*x), -1j*w*self.tr(w,t)*self.Cr(m,k,eps) \
                   *(-1j*m/eps)**2*np.exp(-1j*m/eps*z[self.stratindex[0]])\
                   + 1j*w*self.tl(w,t)*self.Cl(m,k,eps)\
                    *(1j*m/eps)**2*np.exp(+1j*m/eps*z[self.stratindex[0]])))

            self.state['b'] = -self.data/k**2

            return

    def get_fluxes(self):
        '''get derived quantities, like energy and momentum fluxes, from previously calcuated state'''
        self.derived = {}
        self.derived['pwbar'] = self.state['p']*self.state['w'].conjugate()
        self.derived['pubar'] = self.state['p']*self.state['u'].conjugate()
        self.derived['uwbar'] = self.state['u']*self.state['w'].conjugate()
        self.derived['uubar'] = self.state['u']*self.state['u'].conjugate()
        self.derived['pw'] = self.state['p']*self.state['w']
        self.derived['pu'] = self.state['p']*self.state['u']
        self.derived['uw'] = self.state['u']*self.state['w']
        self.derived['uu'] = self.state['u']*self.state['u']


        return

    def calculate_split(self):
        '''split previously calculated state variables into M and m waves and calculate fluxes from individual pieces,
         NOTE: this purposely neglects cross terms in the fluxes, which can be recovered by differencing with the totals.'''

        # get state
        m = self.m
        k = self.k
        w = self.N1*k/m # omega
        t = self.t
        dm = 1.
        dk = 1.

        #calculate just the 'M' terms

        if 'w' in self.state:
            self.reset_data()
            self.data[:,self.tropindex[0]] = self.data[:,self.tropindex[0]] + \
            dm*dk*np.outer(np.exp(1j*k*x),(self.alpha(m,k)*(self.tr(w,t)
                    + self.tl(w,t))*np.sin(self.M*z[self.tropindex[0]])))

            self.state['w_M'] = self.data
            self.state['w_m'] = self.state['w']-self.state['w_M']

                        # calculate partial_z w
            self.reset_data()
            self.data[:,self.tropindex[0]] = self.data[:,self.tropindex[0]] \
                + dm*dk*np.outer(np.exp(1j*k*x),(self.alpha(m,k)*(self.tr(w,t)\
                + self.tl(w,t))*self.M*np.cos(self.M*z[self.tropindex[0]])))

            self.state['u_M'] = self.data
            self.state['u_m'] = self.state['u'] - self.state['u_M']

        if 'p' in self.state:
            self.reset_data()
            self.data[:,self.tropindex[0]] = self.data[:,self.tropindex[0]] \
                + dm*dk*np.outer(np.exp(1j*k*x),(self.alpha(m,k)*(-1j*w*self.tr(w,t)\
                + 1j*w*self.tl(w,t))*self.M*np.cos(self.M*z[self.tropindex[0]])))

            self.state['p_M'] = -self.data/k**2
            self.state['p_m'] = self.state['p'] - self.state['p_M']

            # calculate b = partial_z p
            self.reset_data()
            self.data[:,self.tropindex[0]] = self.data[:,self.tropindex[0]] \
                + dm*dk*np.outer(np.exp(1j*k*x),(self.alpha(m,k)*(-1j*w*self.tr(w,t)\
                + 1j*w*self.tl(w,t))*(-self.M**2)*np.sin(self.M*z[self.tropindex[0]])))

            self.state['b_M'] = -self.data/k**2
            self.state['b_m'] = self.state['b'] -self.state['b_M']

        # some flux terms too
        self.derived['pw_m'] = self.state['p_m']*self.state['w_m'].conjugate()
        self.derived['pw_M'] = self.state['p_M']*self.state['w_M'].conjugate()
        self.derived['pw_cross'] =self.derived['pwbar'] - self.derived['pw_m'] - self.derived['pw_M']

        return
