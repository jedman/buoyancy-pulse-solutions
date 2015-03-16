import numpy as np 

def rigid_modes(Nsq,zsol):
    ''' get vertical modes for stratified boussinesq fluid with a rigid lid;
        d^2/dz^2 w = -N^2 w'''
    zi = zsol[1:-1]
    Nsqgrd = Nsq[1:len(zsol)-1]
    dz = zi[1] - zi[0]

    M = np.zeros((len(zi),len(zi)))

    # apply rigid lid boundary condition
    M[0,0] = 2./(dz**2*Nsqgrd[0])
    M[0,1] = -1./(Nsqgrd[0]*dz**2)
    M[-1,-1] = 2./(dz**2*Nsqgrd[-1])
    M[-1,-2] = -1./(Nsqgrd[-1]*dz**2)
    # fill in the rest of the matrix
    for i in range(1, len(zi)-1):
        M[i,i-1] = -1./(Nsqgrd[i]*dz**2)
        M[i,i] = 2./(dz**2*Nsqgrd[i])
        M[i,i+1] = -1./(Nsqgrd[i]*dz**2)

    c_w, Z_w = np.linalg.eig(M)
    X = zip(c_w,Z_w.transpose())
    X_sort = sorted(X,key=lambda val: val[0]) # sort by eigenvalue
    # sort modes, add top and bottom  interface values, and make them all positive first
    modes_list = []
    for index,mode in enumerate(X_sort):
        tmp = np.zeros(len(mode[1])+2)
        tmp[1:-1] = mode[1]
        #normalize the mode
        tmp = normalize_mode(tmp,Nsq,zsol)
        if tmp[1] > 0:
            modes_list.append(tmp)
        else:
            modes_list.append(-tmp)

    speeds=[]
    trunc = X_sort[0][1].shape[0]
    for i in range(0,trunc):
        speeds.append(1./np.sqrt(X_sort[i][0]))

    return (modes_list, speeds)

def normalize_mode(mode, weight,zsol ):
    '''normalize a mode with respect to weight on grid zsol'''
    coeff = 1/np.sqrt(innerproduct(mode,mode,weight,zsol))
    normed_mode = coeff * mode
    return normed_mode

def innerproduct(mode1, mode2, weight, zsol):
    dz = zsol[1] - zsol[0]
    tot = 0.
    for i in range(0, len(mode1)):
        tot = tot + mode1[i]*mode2[i]*weight[i]*dz
    return tot
