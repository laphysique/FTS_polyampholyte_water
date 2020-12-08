# Dec 8, 2020
# - Field theoretic simulation (FTS) code for an explicit-solvent 
#   polyampholyte solution
# - npoly polyampholytes, nw solvent (water) molecules
# - No dipole moments

import numpy as np

from numpy.linalg import solve 
from numpy.fft import fftn as ft
from numpy.fft import ifftn as ift

import os
import sys

import CL_seq_list as sl

# Bond length = 1
#----------------------- Define polymer solution as a class object -----------------------
class PolySol:
    def __init__( self, lb, sigma, npoly, nw, \
                  v0=0.0068, kappa=0, Nx=24, a=1./np.sqrt(6) ):
        self.lb  = lb                  # Reduced Bjerrum length 
        self.sig = np.array(sigma)     # Charge sequence
        self.N   = self.sig.shape[0]   # Length of polyampholyte
        self.np  = npoly               # Number of polyampholytes 
        self.nw  = nw                  # Number of water molecules
        self.v0  = v0                  # excluded volume    
        self.ksc = kappa               # Debye screening wave number 
        self.a   = a                   # Smearing length
        self.Nx  = Nx                  # Number of grid points: the resolution
        self.L   = a*Nx                # Box edge length
        self.V   = self.L**3           # Box volume
        self.dx  = self.L/Nx           # distance between two n.n. grid points
        self.dV  = self.dx**3          # delta volume of each grid
     
        if np.sum(self.sig) != 0 :
            print('Error: the polymer must be charge neutral.')
            return -1    

        # wave number vectors of the grid space
        ks1d    = 2*np.pi*np.fft.fftfreq( Nx,self.dx ) # k's in 1D reciprocal space
        self.kz = np.tile(ks1d, (Nx, Nx, 1))  # 3D array with kz[i,j,l] = ksld[l]
        self.kx = np.swapaxes( self.kz, 0, 2) # 3D array with kx[i,j,l] = ksld[i]
        self.ky = np.swapaxes( self.kz, 1, 2) # 3D array with ky[i,j,l] = ksld[j]
        self.k2 = self.kx*self.kx + self.ky*self.ky + self.kz*self.kz # 3D array of k*k
         
        self.Gamma   = np.exp(-self.k2*a*a/2)  # Gaussian smearing
        self.Prop    = np.exp(-self.k2/6 )     # Gaussian chain n.n propagator
        self.GT2_w   = ( self.k2*a*a/3 - 1/2 )*self.Gamma  # smearing in pressure
        self.GT2_psi = ( self.k2*a*a/3 - 1/6 )*self.Gamma  # smearing in pressure


        # Gaussian chain correlation functions in the k-space
        gij = np.exp(-np.tensordot(np.arange(self.N), self.k2, axes=0)/6)

        mcc = np.kron(self.sig, self.sig).reshape((self.N, self.N))
        Tcc = np.array([ np.sum(mcc.diagonal(n) + mcc.diagonal(-n)) \
                         for n in range(self.N)]) 
        Tcc[0] /= 2

        Tdd = 2*np.arange(N,0,-1)
        Tdd[0] /= 2   
 
        mcd = np.kron(self.sig, np.ones(self.N)).reshape((self.N, self.N))
        Tcd = np.array([ np.sum(mcd.diagonal(n) + mcd.diagonal(-n)) \
                         for n in range(self.N)])                
        Tcd[0] /= 2

        self.Gcc = gij.T.dot(Tcc).T
        self.Gdd = gij.T.dot(Tdd).T  
        self.Gcd = gij.T.dot(Tcd).T   

    # taking Laplacian of x via Fourier transformation
    def lap(self, x):
        return -ift( self.k2 * ft( x ) ) 

    # Calculate field propagotor in single polymer partition function
    def calc_prop(self, PSI):
        qF = np.zeros( ( self.N, self.Nx, self.Nx, self.Nx ), dtype=complex )
        qB = np.zeros( ( self.N, self.Nx, self.Nx, self.Nx ), dtype=complex )
        qF[0]  = np.exp( -PSI[0]  )
        qB[-1] = np.exp( -PSI[-1] )
    
        for i in range( self.N-1 ):
            # forwards propagator
            qF[i+1] = np.exp( -PSI[i+1] )*ift( self.Prop*ft(qF[i]) )

	    # backwards propagator
            j = self.N-i-1
            qB[j-1] = np.exp( -PSI[j-1] )*ift( self.Prop*ft(qB[j]) )

        return qF, qB

    # Obtain molecule densities from fields
    def calc_densities( self, w, psi, q_output=False ):
        w_s   = ift( self.Gamma*ft(w)  ) 
        psi_s = ift( self.Gamma*ft(psi)  )    

        # Polymer density
        PSI =  1j*( np.tensordot( np.ones(self.N), w_s, axes=0 ) + \
                    np.tensordot( self.sig ,psi_s ,axes=0)           )   
        qF, qB = self.calc_prop(PSI)
        qs     = qF*qB*np.exp(PSI)
        Qp     = np.sum(qF[-1]) * self.dV/ self.V
        rhop   = self.np/self.V/Qp * np.sum(qs, axis=0) 
        rhocp  = self.np/self.V/Qp * qs.T.dot(self.sig).T 

        # solvent
        qw   = np.exp(w_s)
        Qw   = np.sum(qw)*self.dV/self.V
        rhow = self.nw/self.V*qw/Qw  
 
        if q_output:
            return rhop, rhocp, rhow, Qp, Qw, qF, qB, qw  
        else:
            return rhop, rhocp, rhow

#---------------------------- Complex Langevin Time Evolution ----------------------------

# Semi-implicit method 
def CL_step_SI(w, psi, PS, M_inv, dt, useSI=True):

    rhop, rhocp, rhow = PS.calc_densities(w, psi)

    std     = np.sqrt( 2 * dt / PS.dV )
    eta_w   = std*np.random.randn( PS.Nx, PS.Nx, PS.Nx )
    eta_psi = std*np.random.randn( PS.Nx, PS.Nx, PS.Nx ) 
 
    dw   = -dt*( 1j*ift( PS.Gamma*ft( rhop+rhow ) ) + w/PS.v0 ) + eta_w
    dpsi = -dt*( 1j*ift( PS.Gamma*ft( rhocp ) ) \
                 + (PS.ksc*PS.ksc*psi- PS.lap(psi))/(4*np.pi*PS.lb) ) + eta_psi

    # Semi-implicit CL step
    if useSI:
        ftdw, ftdpsi = ft( dw ) , ft( dpsi )
        dw_tmp   = M_inv[0,0] * ftdw + M_inv[0,1] * ftdpsi
        dpsi_tmp = M_inv[1,0] * ftdw + M_inv[1,1] * ftdpsi
 
        w   += ift( dw_tmp ) 
        psi += ift( dpsi_tmp ) 
    else:
        w   += dw
        psi += dpsi

    w   -= np.mean(w) + 1j*(PS.np*PS.N+PS.nw)/(PS.V*PS.v0) 
    psi -= np.mean(psi)

    return w , psi
 
# get M_inv for semi-implicit CL integration method
def get_M_inv( PS, dt):
    K11 = PS.Gamma*PS.Gamma* ( PS.np*PS.Gdd + PS.nw)/PS.V
    K12 = PS.Gamma*PS.Gamma* PS.np*PS.Gcd/PS.V
    K22 = PS.Gamma*PS.Gamma* PS.np*PS.Gcc/PS.V + PS.k2 / (4*np.pi*PS.lb)
    K11[0,0,0] = 0
  
    M = np.array( [ [ 1+dt*K11 , dt*K12 ] , [ dt*K12 , 1+dt*K22 ] ]  )
    det_M = M[0,0] * M[1,1] - M[0,1] * M[1,0]
    M_inv = np.array( [ [ M[1,1] , - M[0,1] ] , [ - M[1,0] , M[0,0] ] ] ) / det_M

    return M_inv

#------------------------------- Thermodynamic quantities --------------------------------

# Chemical potentials
def get_chem_potential( w, psi, PS ):
    rhop, rhocp, rhow, Qp, Qw, _, _, _ = PS.calc_densities(w, psi, q_output=True )

    mu_p = np.log( PS.np ) + 1 - np.log( Qp) 
    mu_w = np.log( PS.nw ) + 1 - np.log( Qw) 
    

    return mu_p, mu_w

# Osmotic pressure
def get_pressure( w, psi, PS ):
    rhop, rhocp, rhow, Qp, Qw, qF,qB, _ = PS.calc_densities(w, psi, q_output=True )
 
    # fluctuation
    ftw , ftpsi = ft(w), ft(psi)

    w_s   = ift(PS.Gamma*ftw) 
    psi_s = ift(PS.Gamma*ftpsi) 
    
    PSI = 1j*( np.tensordot( np.ones(PS.N), w_s, axes=0 ) + \
               np.tensordot( PS.sig ,psi_s ,axes=0)           )   

    lap_qF =  np.array([  PS.lap( np.exp(PSI[i])*qF[i] ) for i in range(PS.N)] )
    term1 = np.sum(np/(9*Qp) * qB*lap_qF)
    term2 = 1j*np.sum( (rhop+rhow)*ift( PS.GT2_w*ftw ) )
    term3 = 1j*np.sum( rhocp*ift( PS.GT2_psi*ftpsi )   )

    return (PS.np + PS.nw - PS.nw*np.log(Qw) - (term1+term2+term3)*PS.dV)/PS.V

    

#-------------------------------- Input/Output functions ---------------------------------

# Write to file:
def save_a_snapshot(w, psi, PS, seqname, istep, dt, dirname=None):
    par_info = '_' + seqname + \
               '_lb'   + str(PS.lb) + \
               '_np'   + str(int(PS.np)) + \
               '_nw'   + str(int(PS.nw)) + \
               '_v'    + str(PS.v0) + \
               '_kscr' + str(PS.ksc) + \
               '_a'    + str(PS.a) + \
               '_Nx'   + str(int(PS.Nx)) + \
               '_dt'   + str(dt) 
    
    if dirname==None:
        dirname = './results/All_steps_of' + par_info 

    if os.path.isdir(dirname):
        pass
    else:
        os.makedirs(dirname) 
               
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

    fname = str(int(istep)) + 'th_step_of' + parinfo 
    fmt   = ' '.join(['%i']*3 + ['%.8f']*4)
    hdr   = '   x      y      z   ' + \
            '     Re[w]          Im[w]          Re[psi]          Im[psi]'  

    np.savetxt( fname, fields, fmt=fmt, header=hdr)

    return par_info, dirname

# Read from file
def read_a_snapshos( info=None, dir_and_file=None, make_PS=False ):
    if info is not None:
        istep, lb, npoly, nw, v, ksc, a, Nx, dt = info
  
        par_info = '_' + seqname + \
                   '_lb'   + str(lb) + \
                   '_np'   + str(int(npoly)) + \
                   '_nw'   + str(int(nw)) + \
                   '_v'    + str(v) + \
                   '_kscr' + str(ksc) + \
                   '_a'    + str(a) + \
                   '_Nx'   + str(int(Nx)) + \
                   '_dt'   + str(dt) 

    if dirname==None:
        dirname = './results/All_steps_in' + par_info 
        fname   = str(int(istep)) + 'th_step_of' + parinfo + '.txt'
    elif dir_and_file is not None:
        dirname, fname = dir_and_file  
        istep = int( fname[0:fname.index('th')] )
        all_info = dirname[dirname.index('_lb')+3:].split('_')        
        lb    = float(all_info[0])
        npoly = int(  all_info[1][2:])
        nw    = int(  all_info[2][2:]) 
        v     = float(all_info[3][1:]) 
        ksc   = float(all_info[4][4:])  
        #a    = float(all_info[5][1:])  
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
        PS = PolySol(lb, sig, npoly, nw, v0=v, kappa=ksc, Nx=Nx)   
 
        return w, psi, dt, istep, PS
   
    else:
        return w, psi, dt, istep


#------------------------------------- Main function -------------------------------------

if __name__ == "__main__":
  
    # CL time step
    dt = 0.001            # time interval in simulation
    nT = 100000           # total number of time steps
    t_prod = int(0.5*nT)  # sampling when t > t_prod
    dT_snapshot = 10      # time interval between two snapshots

    # charge sequence
    seqname = sys.argv[1]
    sig, N, the_seq= sl.get_the_charge(seqname)
    print(the_seq, N)
    lb = 2
    npoly = 20
    nw    = 0    
 
    # polymer solution object
    PS = PolySol(lb, sig, npoly, nw)
    par_info = '_' + seqname + \
               '_lb'   + str(PS.lb) + \
               '_np'   + str(int(PS.np)) + \
               '_nw'   + str(int(PS.nw)) + \
               '_v'    + str(PS.v0) + \
               '_kscr' + str(PS.ksc) + \
               '_a'    + str(PS.a) + \
               '_Nx'   + str(int(PS.Nx)) + \
               '_dt'   + str(dt) 


    # initialization
    w   = 0.001*np.random.randn( PS.Nx,PS.Nx,PS.Nx ) + \
          1j*0.001*np.random.randn( PS.Nx,PS.Nx,PS.Nx )
    psi = 0.001*np.random.randn( PS.Nx,PS.Nx,PS.Nx ) + \
          1j*0.001*np.random.randn( PS.Nx,PS.Nx,PS.Nx )

    w   -= np.mean(w) + 1j*(PS.np*PS.N+PS.nw)/(PS.V*PS.v0) 
    psi -= np.mean(psi)

    Minv = get_M_inv( PS, dt)
    for t in range(t_prod):
        #print(t, np.max(np.abs(w)), np.max(np.abs(psi)), np.mean(w), np.mean(psi)) 
        CL_step_SI(w, psi, PS, Minv, dt, useSI=False)        
       
    muPI = open('muPI' + par_info + '.txt', 'w')
    muPI.write( '# mu PI' )   
 
    for t in range(t_prod, nT):
        if t % dT_snapshot == 0:
            save_a_snapshot(w, psi, PS, seqname, t, dt)
            mu = get_chem_potential(w, psi, PS)
            PI = get_pressure(w, psi, PS)
            muPI.write('{:.8f} {:.8f}'.format(mu, PI)  )                 
    
        CL_step_SI(w, psi, PS, Minv, dt) 
         
