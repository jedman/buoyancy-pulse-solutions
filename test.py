import matplotlib.pyplot as plt 
import numpy as np 
from PulseClass import * 


z = np.linspace(0,50000,200)
Nsq = np.zeros(len(z))
x = np.linspace(-10000,10000,200)
H= 16000.
t = np.linspace(0,1*86400, 2400) # time 
tropoindex = np.where(z < H) 
# define the stability profile
Nsq[:] = 0.0004
Nsq[tropoindex] = 0.0001
zt = z[tropoindex]

test = PulseSolution(x,z,z,t,Nsq, method = 'rigid', domain = 'periodic')

test.get_vertical_modes()

inic = np.zeros(len(z)) # define initial condition
tropoindex = np.where(z <= H)
inic[tropoindex] = np.sin(np.pi*z/H)

test.get_solution(inic, 0,100, solvefor= 'buoyancy')

plt.ion() 

plt.pcolormesh(test.x, test.z, test.data_b[0,:,:]) 
plt.colorbar() 
