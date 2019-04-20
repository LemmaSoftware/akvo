import numpy as np 
from scipy.optimize import least_squares 
from scipy.optimize import minimize
from scipy.linalg import lstsq as sclstsq

def harmonic2 ( f1, f2, sN, fs, nK, t ): 
    """
    Performs inverse calculation of two harmonics contaminating a signal. 
    Args:
        f01 = base frequency of the first sinusoidal noise 
        f02 = base frequency of the second sinusoidal noise 
        sN = signal containing noise 
        fs = sampling frequency
        nK = number of harmonics to calculate 
        t = time samples 
    """
    print("building matrix ")
    A = np.zeros( (len(t),  4*nK) )
    #f1 = f1MHz * 1e-3
    #f2 = f2MHz * 1e-3
    for irow, tt in enumerate(t): 
        A[irow, 0:2*nK:2] = np.cos( np.arange(nK)*2*np.pi*(f1/fs)*irow )
        A[irow, 1:2*nK:2] = np.sin( np.arange(nK)*2*np.pi*(f1/fs)*irow )
        A[irow, 2*nK::2] = np.cos( np.arange(nK)*2*np.pi*(f2/fs)*irow )
        A[irow, 2*nK+1::2] = np.sin( np.arange(nK)*2*np.pi*(f2/fs)*irow )

    v = np.linalg.lstsq(A, sN, rcond=1e-8)
    #v = sclstsq(A, sN) #, rcond=1e-6)

    alpha = v[0][0:2*nK:2]
    beta  = v[0][1:2*nK:2]
    amp = np.sqrt( alpha**2 + beta**2 )
    phase = np.arctan(- beta/alpha)
    
    alpha2 = v[0][2*nK::2]
    beta2  = v[0][2*nK+1::2]
    amp2 = np.sqrt( alpha2**2 + beta2**2 )
    phase2 = np.arctan(- beta2/alpha2)

    h = np.zeros(len(t))
    for ik in range(nK):
        h += np.sqrt(alpha[ik]**2 + beta[ik]**2) * np.cos( 2.*np.pi*ik * (f1/fs) * np.arange(0, len(t), 1 )  + phase[ik] ) \
           + np.sqrt(alpha2[ik]**2 + beta2[ik]**2) * np.cos( 2.*np.pi*ik * (f2/fs) * np.arange(0, len(t), 1 )  + phase2[ik] )

    return sN-h

def harmonic ( f0, sN, fs, nK, t ): 
    """
    Performs inverse calculation of harmonics contaminating a signal. 
    Args:
        f0 = base frequency of the sinusoidal noise 
        sN = signal containing noise 
        fs = sampling frequency
        nK = number of harmonics to calculate 
        t = time samples 
    """
    print("building matrix ")
    A = np.zeros( (len(t),  2*nK) )
    for irow, tt in enumerate(t): 
        A[irow, 0::2] = np.cos( np.arange(nK)*2*np.pi*(f0/fs)*irow )
        A[irow, 1::2] = np.sin( np.arange(nK)*2*np.pi*(f0/fs)*irow )

    v = np.linalg.lstsq(A, sN, rcond=None) #, rcond=1e-8)

    alpha = v[0][0::2]
    beta  = v[0][1::2]

    amp = np.sqrt( alpha**2 + beta**2 )
    phase = np.arctan(- beta/alpha)

    #print("amp:", amp, " phase", phase)

    h = np.zeros(len(t))
    for ik in range(nK):
        h += np.sqrt(alpha[ik]**2 + beta[ik]**2) * np.cos( 2.*np.pi*ik * (f0/fs) * np.arange(0, len(t), 1 )  + phase[ik] )

    #plt.matshow(A, aspect='auto')
    #plt.colorbar()

    #plt.figure()
    #plt.plot(alpha)
    #plt.plot(beta)
    #plt.plot(amp)

    #plt.figure()
    #plt.plot(h)
    #plt.title("modelled noise")
    return sN-h

def jacobian( f0, sN, fs, nK, t):
    print("building Jacobian matrix ")
    A = np.zeros( (len(t),  2*nK) )
    for irow, tt in enumerate(t): 
        #A[irow, 0::2] = np.cos( np.arange(nK)*2*np.pi*(f0/fs)*irow )
        #A[irow, 1::2] = np.sin( np.arange(nK)*2*np.pi*(f0/fs)*irow )
        # brutal 
        for k, ik in enumerate( np.arange(0, 2*nK, 2) ):
            #A[irow, ik  ] = np.cos( k*2*np.pi*(f0/fs)*irow )
            #A[irow, ik+1] = np.sin( k*2*np.pi*(f0/fs)*irow )    
            A[irow, ik  ] = - (2.*np.pi*k*irow * sin((2.*np.pi*irow*f0)/fs)) / fs
            A[irow, ik+1] =   (2.*np.pi*k*irow * cos((2.*np.pi*irow*f0)/fs)) / fs


def harmonicNorm ( f0, sN, fs, nK, t ): 
    return np.linalg.norm( harmonic(f0, sN, fs, nK, t))

def harmonic2Norm ( f0, sN, fs, nK, t ): 
    return np.linalg.norm(harmonic2(f0[0], f0[1], sN, fs, nK, t))

def minHarmonic(f0, sN, fs, nK, t):
    f02 = guessf0(sN, fs)
    print("minHarmonic", f0, fs, nK, " guess=", f02)
    res = minimize( harmonicNorm, np.array((f0)), args=(sN, fs, nK, t)) #, method='Nelder-Mead' )# jac=None, hess=None, bounds=None )
    print(res)
    return harmonic(res.x[0], sN, fs, nK, t)

def minHarmonic2(f1, f2, sN, fs, nK, t):
    #f02 = guessf0(sN, fs)
    #print("minHarmonic2", f0, fs, nK, " guess=", f02)
    #methods with bounds, L-BFGS-B, TNC, SLSQP
    res = minimize( harmonic2Norm, np.array((f1,f2)), args=(sN, fs, nK, t)) #, bounds=((f1-1.,f1+1.0),(f2-1.0,f2+1.0)), method='SLSQP' )
    print(res)
    return harmonic2(res.x[0], res.x[1], sN, fs, nK, t) 

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
    delta  = 0 #np.random.rand() 
    delta2 = 0 #np.random.rand() 
    print("delta", delta)
    fs = 10000   # GMR 
    t = np.arange(0, 1, 1/fs)
    phi = 0 #np.random.rand() 
    phi2 = 0 # np.random.rand() 
    A =  1.0
    A2 = 0.0 
    nK = 10
    T2 = .200
    sN  = A * np.sin( ( 1*(delta  +f0))*2*np.pi*t + phi ) + \
          A2* np.sin( ( 1*(delta2 +f1))*2*np.pi*t + phi2 ) + \
              np.random.normal(0,.1,len(t)) + \
              + np.exp( -t/T2  ) 

    sNc = A * np.sin(  (1*(delta +f0))*2*np.pi*t + phi ) + \
          A2* np.sin(  (1*(delta2+f1))*2*np.pi*t + phi2 ) + \
              + np.exp( -t/T2  ) 


    guessf0(sN, fs)

    #h = harmonic( f0, sN, fs, nK, t) 
    #h = minHarmonic2( f0, f1, sN, fs, nK, t) 
    h = harmonic2( f0, f1, sN, fs, nK, t) 

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

