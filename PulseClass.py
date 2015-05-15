import numpy as np
import rigidmodes as rm 
class PulseSolution:
    ''' class for solutions to initial buoyancy perturbation in a boussinesq fluid'''
    def __init__(self,x,zdom,zsol,time,Nsq, method = 'rigid', domain = 'infinite'):
        if method not in ('rigid', 'leaky'): 
            raise Exception, method +'is not a valid solution method!'
        if method == 'leaky': 
            self.Nrat = np.sqrt(Nsq[0]/Nsq[-1])
        self.method = method
        self.x = x 
        self.z = zdom #full domain
        self.zsol = zsol # solution domain
        self.time = time 
        self.Nsq = Nsq[0:len(zsol)] 
        self.data_b = np.zeros((len(time),len(zdom), len(x))) # buoyancy data array
        self.data_w = np.zeros((len(time),len(zdom), len(x)))
        if domain not in ('infinite', 'periodic'):
            raise Exception, self.domain + 'is not a valid domain type' 
        self.domain = domain
        self.width = 2.*max(x)
        self.extent = (min(x), max(x)) 
        
    def get_vertical_modes(self):
        if self.method == 'rigid':
            self.modes, self.speeds =  rm.rigid_modes(self.Nsq, self.zsol)
       # elif self.method == 'leaky':
    
    def horiz_dep(self, y , a = 1000. ):
        '''get the horizontal part of the solution. assumes initial b(x) = 
           a**2/(a**2 + x**2).''' 
        if self.domain == 'infinite':
            widthfun = a**2/(y**2 + a**2) 
        if self.domain == 'periodic':
            if y < self.extent[0]:
                while y < self.extent[0]:
                    y = y + self.width
            elif y > self.extent[1]:
                while y > self.extent[1]:
                    y = y - self.width
            widthfun = a**2/(y**2 + a**2)             
        return widthfun 
    
    
    def get_solution(self, initial_b , tstep, nmodes, solvefor = 'buoyancy'): 
        if self.method == 'rigid':
            self.initial_b = initial_b # vertical structure of initial buoyancy 
            self.initial_h = -1./self.Nsq[0:len(self.zsol)] * initial_b[0:len(self.zsol)]
            self.data = self.rigidlid_solution(tstep, nmodes)
            if  'buoyancy' in solvefor:
                Nsqmat = np.zeros((len(self.z), len(self.x)))
                for i in range(0, len(self.x)):
                    Nsqmat[0:len(self.zsol),i] = self.Nsq
                self.data_b[tstep, :,:]= -Nsqmat*self.data
            if 'w' in solvefor:
                tmp = self.rigidlid_solution(tstep-1, nmodes)
                self.data_w[tstep,:,:] = (self.data - tmp)/(self.time[tstep]- self.time[tstep-1])
            #if ('buoyancy' and 'w')  not in solvefor:
            #    raise Exception, 'solution type '+ solvefor[:] + ' not implemented'
                
    def rigidlid_solution(self, tstep, nmodes): 
        #rho0 = 1.0
        #rho0 = np.exp(-self.Nsq/9.8*self.zsol)
        weight = self.Nsq
        self.weight = weight
        coeffs = np.zeros(nmodes)
        for i in range(0,nmodes):
            coeffs[i] = rm.innerproduct(self.initial_h, self.modes[i],weight, self.zsol)

        m_x = [] 
        m_tmp = np.zeros((len(self.zsol), len(self.x)))
 
        for j in range(nmodes):
           for i in range(len(self.x)): 
              m_tmp[:,i] = 0.5*self.horiz_dep(self.x[i]-self.speeds[j]*self.time[tstep])*self.modes[j]\
              +0.5*self.horiz_dep(self.x[i]+self.speeds[j]*self.time[tstep])*self.modes[j]    
           m_x.append(coeffs[j]*m_tmp.copy())
    
        superproject = np.zeros((len(self.zsol), len(self.x))) 
        for index,mode in enumerate(m_x):
            superproject = superproject + mode
            
        fullsolution = np.zeros((len(self.z), len(self.x)))
        fullsolution[0:len(self.zsol), :] = superproject

        return fullsolution

