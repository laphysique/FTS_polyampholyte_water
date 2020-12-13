# ver1 Dec 10, 2020
# - The input/output functions supporting FTS_polyampholyte_water.py 

import sys
import os
import numpy as np
import multiprocessing as mp
import CL_seq_list as sl
import FTS_polyampholyte_water as fts

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
    fmt   = ' '.join(['%3d']*3 + ['%16.8e']*4)
    hdr   = 'x   y   z ' + \
            '       Re[w]            Im[w]          Re[psi]          Im[psi]'  
    
    hdr += '\n' + '='*(len(hdr)+6)

    np.savetxt( dirname + '/' + fname, fields, fmt=fmt, header=hdr)

    return par_info, dirname

# Read from file
def read_a_snapshot( info=None, dir_and_file=None, make_PS=False ):
    if info is not None:
        istep, seqname, lb, npoly, nw, v, ksc, invasq, Nx, dt = info
  
        par_info = '_' + seqname + \
                   '_lb'   + str(lb) + \
                   '_np'   + str(int(npoly)) + \
                   '_nw'   + str(int(nw)) + \
                   '_v'    + str(v) + \
                   '_kscr' + str(ksc) + \
                   '_invasq' + str(invasq) + \
                   '_Nx'   + str(int(Nx)) + \
                   '_dt'   + str(dt) 
    
        dirname = './results/All_steps_in' + par_info 
        fname   = str(int(istep)) + 'th_step_of' + par_info + '.txt'
    elif dir_and_file is not None:
        dirname, fname = dir_and_file  
        
        seqname = dirname[ dirname.index('_of_')+4:dirname.index('_lb') ]
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
    w   = (PSI[:,3] + 1j*PSI[:,4]).reshape((Nx,Nx,Nx))
    psi = (PSI[:,5] + 1j*PSI[:,6]).reshape((Nx,Nx,Nx))

    if make_PS:
        sig, _, _ = sl.get_the_charge(seqname)
        PS = fts.PolySol(lb, sig, npoly, nw, v0=v, kappa=ksc, inva2=inva2, Nx=Nx)   
 
        return w, psi, dt, istep, PS
   
    else:
        return w, psi, dt, istep


# Parallelly read fields and calculate densities
def get_mean_densities(dirfilePS):
    w, psi, _, istep , = read_a_snapshot( dir_and_file=(dirfilePS[0], dirfilePS[1]) )
    rhop, rhocp, rhow = dirfilePS[2].calc_densities( w, psi )
    print(str(istep) + ' done!', flush=True )
    return rhop, rhocp, rhow
   

def read_all_snapshots( dirname, res_to_file=False ):

    print('dirname:',dirname)
    
    seqname = dirname[ dirname.index('_of_')+4:dirname.index('_lb') ]

    dir_info = dirname[dirname.index('_lb')+3:].split('_')        
    lb    = float(dir_info[0])
    npoly = int(  dir_info[1][2:])
    nw    = int(  dir_info[2][2:]) 
    v     = float(dir_info[3][1:]) 
    ksc   = float(dir_info[4][4:])  
    inva2 = float(dir_info[5][6:])  
    Nx    = int(  dir_info[6][2:])
    dt    = float(dir_info[7][2:])
    
    sig, _, _ = sl.get_the_charge(seqname)
    PS = fts.PolySol(lb, sig, npoly, nw, v0=v, kappa=ksc, inva2=inva2, Nx=Nx)   

    all_snapshots = os.listdir(dirname)
    print('number of files:', len(all_snapshots))

    dirfiles = [ (dirname, s, PS) for s in all_snapshots   ] 

    pool = mp.Pool(processes=40)
    rhop_all, rhocp_all, rhow_all = zip(*pool.map( get_mean_densities, dirfiles ))

    rhop_avg  = np.mean(rhop_all, axis=0)
    rhocp_avg = np.mean(rhocp_all, axis=0)
    rhow_avg  = np.mean(rhow_all, axis=0)

    isteps = []
    for i, fname in enumerate(all_snapshots):
        isteps.append( int(fname[:fname.index('th_step')]) )
    if res_to_file:
        finfo = '_mean_steps' + str(np.min(isteps)) + 'to' + str(np.max(isteps)) + '.npy' 
        np.save( 'rhop'  + finfo , rhop_avg ) 
        np.save( 'rhocp' + finfo , rhocp_avg) 
        np.save( 'rhow'  + finfo , rhow_avg ) 

    return rhop_avg, rhocp_avg, rhow_avg 


if __name__ == "__main__":
    read_all_snapshots( sys.argv[1], res_to_file=True )
