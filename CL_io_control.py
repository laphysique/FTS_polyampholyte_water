# ver1 Dec 10, 2020
# - The input/output functions supporting FTS_polyampholyte_water.py 

import sys
import os
import numpy as np

# Write to file:
def save_a_snapshot(w, psi, PS, seqname, istep, dt, dirname=None):
    par_info = '_' + seqname + \
               '_lb'     + str(PS.lb) + \
               '_np'     + str(int(PS.np)) + \
               '_nw'     + str(int(PS.nw)) + \
               '_v'      + str(PS.v0) + \
               '_kscr'   + str(PS.ksc) + \
               '_invasq' + str(round(1/PS.a**2)) + \
               '_Nx'     + str(int(PS.Nx)) + \
               '_dt'     + str(dt) 
    
    if dirname==None:
        dirname = './results/All_steps_of' + par_info 

    if os.path.isdir(dirname):
        pass
    else:
        os.makedirs(dirname) 
    

    Nx = PS.Nx           
    fields = np.zeros(( Nx*Nx*Nx, 7 ))
    for i in range(Nx):
        for j in range(Nx):
            for k in range(Nx):
                ii =  Nx*Nx*i + Nx*j + k
                fields[ ii, 0:3] = [i, j, k]
                fields[ ii, 3 ]  = w[i,j,k].real
                fields[ ii, 4 ]  = w[i,j,k].imag  
                fields[ ii, 5 ]  = psi[i,j,k].real
                fields[ ii, 6 ]  = psi[i,j,k].imag  

    fname = str(int(istep)) + 'th_step_of' + par_info + '.txt'
    fmt   = ' '.join(['2%d']*3 + ['%.8e']*4)
    hdr   = ' x  y  z ' + \
            '    Re[w]        Im[w]        Re[psi]        Im[psi]'  

    np.savetxt( dirname + '/' + fname, fields, fmt=fmt, header=hdr)

    return par_info, dirname

# Read from file
def read_a_snapshos( info=None, dir_and_file=None, make_PS=False ):
    if info is not None:
        istep, lb, npoly, nw, v, ksc, invasq, Nx, dt = info
  
        par_info = '_' + seqname + \
                   '_lb'   + str(lb) + \
                   '_np'   + str(int(npoly)) + \
                   '_nw'   + str(int(nw)) + \
                   '_v'    + str(v) + \
                   '_kscr' + str(ksc) + \
                   '_invasq' + str(invasq) + \
                   '_Nx'   + str(int(Nx)) + \
                   '_dt'   + str(dt) 

    if dirname==None:
        dirname = './results/All_steps_in' + par_info 
        fname   = str(int(istep)) + 'th_step_of' + par_info + '.txt'
    elif dir_and_file is not None:
        dirname, fname = dir_and_file  
        istep = int( fname[0:fname.index('th')] )
        all_info = dirname[dirname.index('_lb')+3:].split('_')        
        lb    = float(all_info[0])
        npoly = int(  all_info[1][2:])
        nw    = int(  all_info[2][2:]) 
        v     = float(all_info[3][1:]) 
        ksc   = float(all_info[4][4:])  
        inva2 = float(all_info[5][6:])  
        Nx    = int(  all_info[6][2:])
        dt    = float(all_info[7][2:])

    else:
        print('Error: invalid file name')
        return -1        


    PSI = np.loadtxt(dirname + '/' + fname)
    w   = (PSI[:,3] + ij*PSI[:,4]).reshape((Nx,Nx,Nx))
    psi = (PSI[:,5] + ij*PSI[:,6]).reshape((Nx,Nx,Nx))

    if make_PS:
        sig = sl.get_the_charge(seqname)
        PS = PolySol(lb, sig, npoly, nw, v0=v, kappa=ksc, inva2=inva2, Nx=Nx)   
 
        return w, psi, dt, istep, PS
   
    else:
        return w, psi, dt, istep


