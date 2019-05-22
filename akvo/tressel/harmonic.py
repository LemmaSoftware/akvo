import numpy as np 
from scipy.optimize import least_squares 
from scipy.optimize import minimize
from scipy.linalg import lstsq as sclstsq
import scipy.linalg as lin

def harmonicEuler ( f0, sN, fs, nK, t ): 
    """
    Performs inverse calculation of harmonics contaminating a signal. 
    Args:
        f0 = base frequency of the sinusoidal noise 
        sN = signal containing noise 
        fs = sampling frequency
        nK = number of harmonics to calculate 
        t = time samples 
    """
    
    A = np.exp(1j* np.tile( np.arange(1,nK+1),(len(t), 1)) * 2*np.pi* (f0/fs) * np.tile(np.arange(len(t)),(nK,1)).T  )

    v = np.linalg.lstsq(A, sN, rcond=None) 
    alpha = np.real(v[0]) 
    beta  = np.imag(v[0]) 

    amp = np.abs(v[0])     
    phase = np.angle(v[0]) 

    h = np.zeros(len(t))
    for ik in range(nK):
        h +=  2*amp[ik] * np.cos( 2.*np.pi*(ik+1) * (f0/fs) * np.arange(0, len(t), 1 )  + phase[ik] )
    
    return sN-h
    
    res = sN-h # residual 

    # calculate jacobian 
    #J = jacEuler(f0, sN, fs, nK, t)
    #plt.matshow(np.real(J), aspect='auto')
    
    #Jv = J * np.tile(v[0], (len(t) ,1)) 
    #plt.matshow(np.real(Jv), aspect='auto')
    
    #print ("shape J",  np.shape(J))
    #print ("shape v",  np.shape(v[0]))
    #print ("shape Jv",  np.shape(Jv))

    #plt.figure()
    #plt.plot(v[0])
    #plt.figure()
    #plt.plot(Jv)
    #plt.show()
    #exit()

    #jac =  np.linalg.norm( np.dot(Jv.T, res)) 
    #print("norm ||", jac , "||", " at freq", f0)

    #return res, jac 


def jacEuler( f0, sN, fs, nK, t):
    print("building Jacobian matrix ")
    J = np.zeros( (len(t),  nK), dtype=np.complex64 )
    for it, tt in enumerate(t):
        for ik, k in enumerate( np.arange(0, nK) ):
            c = 1j*2.*np.pi*(ik+1.)
            J[it, ik] = (c*np.exp(it*c*f0/fs)) / fs 
    return J

def harmonicEuler2 ( f0, f1, sN, fs, nK, t ): 
    """
    Performs inverse calculation of harmonics contaminating a signal. 
    Args:
        f0 = base frequency of the sinusoidal noise 
        sN = signal containing noise 
        fs = sampling frequency
        nK = number of harmonics to calculate 
        t = time samples 
    """
    A1 = np.exp(1j* np.tile( np.arange(1,nK+1),(len(t), 1)) * 2*np.pi* (f0/fs) * np.tile(np.arange(len(t)),(nK,1)).T  )
    A2 = np.exp(1j* np.tile( np.arange(1,nK+1),(len(t), 1)) * 2*np.pi* (f1/fs) * np.tile(np.arange(len(t)),(nK,1)).T  )
    A = np.concatenate( (A1, A2), axis=1 )


    v = np.linalg.lstsq(A, sN, rcond=None) # rcond=None) #, rcond=1e-8)
    amp = np.abs(v[0][0:nK])     
    phase = np.angle(v[0][0:nK]) 
    amp1 = np.abs(v[0][nK:2*nK])     
    phase1 = np.angle(v[0][nK:2*nK]) 

    h = np.zeros(len(t))
    for ik in range(nK):
        h +=  2*amp[ik]  * np.cos( 2.*np.pi*(ik+1) * (f0/fs) * np.arange(0, len(t), 1 )  + phase[ik] ) + \
              2*amp1[ik] * np.cos( 2.*np.pi*(ik+1) * (f1/fs) * np.arange(0, len(t), 1 )  + phase1[ik] )

    return sN-h

def harmonicNorm ( f0, sN, fs, nK, t ): 
    return np.linalg.norm( harmonicEuler(f0, sN, fs, nK, t) )

def harmonic2Norm ( f0, sN, fs, nK, t ): 
    return np.linalg.norm(harmonicEuler2(f0[0], f0[1], sN, fs, nK, t))

def minHarmonic(f0, sN, fs, nK, t):
    f02 = guessf0(sN, fs)
    print("minHarmonic", f0, fs, nK, " guess=", f02)
    # CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr
    res = minimize(harmonicNorm, np.array((f0)), args=(sN, fs, nK, t), jac='2-point', method='BFGS') #, jac=jacEuler) #, hess=None, bounds=None )
    print(res)
    return harmonicEuler(res.x[0], sN, fs, nK, t)#[0]

def minHarmonic2(f1, f2, sN, fs, nK, t):
    #f02 = guessf0(sN, fs)
    #print("minHarmonic2", f0, fs, nK, " guess=", f02)
    #methods with bounds, L-BFGS-B, TNC, SLSQP
    res = minimize( harmonic2Norm, np.array((f1,f2)), args=(sN, fs, nK, t), jac='2-point', method='BFGS') #, bounds=((f1-1.,f1+1.0),(f2-1.0,f2+1.0)), method='TNC' )
    print(res)
    return harmonicEuler2(res.x[0], res.x[1], sN, fs, nK, t) 

def guessf0( sN, fs ):
    S = np.fft.fft(sN)
    w = np.fft.fftfreq( len(sN), 1/fs )
    imax = np.argmax( np.abs(S) )
    #plt.plot( w, np.abs(S) )
    #plt.show()
    #print(w)
    #print ( w[imax], w[imax+1] )
    return abs(w[imax])

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt 

    f0 = 60      # Hz
    f1 = 60      # Hz
    delta  = np.random.rand() - .5 
    delta2 = np.random.rand() - .5 
    print("delta", delta)
    print("delta2", delta2)
    fs = 10000   # GMR 
    t = np.arange(0, 1, 1/fs)
    phi =  2.*np.pi*np.random.rand() - np.pi 
    phi2 = 2.*np.pi*np.random.rand() - np.pi
    print("phi", phi, phi2)
    A =  1.0
    A2 = 1.0 
    A3 = 1.0 
    nK = 10
    T2 = .200
    sN  = A *np.sin( ( 1*(delta  +f0))*2*np.pi*t + phi ) + \
          A2*np.sin( ( 1*(delta2 +f1))*2*np.pi*t + phi2 ) + \
              np.random.normal(0,.1,len(t)) + \
              + A3*np.exp( -t/T2  ) 

    sNc = A *np.sin(  (1*(delta +f0))*2*np.pi*t + phi ) + \
          A2*np.sin(  (1*(delta2+f1))*2*np.pi*t + phi2 ) + \
              + A3*np.exp( -t/T2  ) 


    guessf0(sN, fs)

    # single freq
    #h = harmonicEuler( f0, sN, fs, nK, t) 
    #h = minHarmonic( f0, sN, fs, nK, t) 
    
    # two freqs 
    h = minHarmonic2( f0+1e-2, f1-1e-2, sN, fs, nK, t) 
    #h = harmonicEuler2( f0, f1, sN, fs, nK, t) 

    plt.figure()
    plt.plot(t, sN, label="sN")
    #plt.plot(t, sN-h, label="sN-h")
    plt.plot(t, h, label='h')
    plt.title("harmonic")
    plt.legend()

    plt.figure()
    plt.plot(t, sN-sNc, label='true noise')
    plt.plot(t, h, label='harmonic removal')
    plt.plot(t, np.exp(-t/T2), label="nmr")
    plt.legend()
    plt.title("true noise")
    
    plt.show()

