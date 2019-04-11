import numpy as np 
from scipy.optimize import least_squares 

def harmonic ( sN, f0, fs, nK, t  ): 
    """
    Performs inverse calculation of harmonics contaminating a signal. 
    Args:
        sN = signal containing noise 
        f0 = base frequency of the sinusoidal noise 
        fs = sampling frequency
        nK = number of harmonics to calculate 
        t = time samples 
    """
    print("building matrix ")
    A = np.zeros( (len(t),  2*nK) )
    for irow, tt in enumerate(t): 
        A[irow, 0::2] = np.cos( np.arange(nK)*2*np.pi*(f0/fs)*irow )
        A[irow, 1::2] = np.sin( np.arange(nK)*2*np.pi*(f0/fs)*irow )
        # brutal 
        #for k, ik in enumerate( np.arange(0, 2*nK, 2) ):
        #    A[irow, ik  ] = np.cos( k*2*np.pi*(f0/fs)*irow )
        #    A[irow, ik+1] = np.sin( k*2*np.pi*(f0/fs)*irow )

    v = np.linalg.lstsq(A, sN, rcond=None) #, rcond=1e-8)

    alpha = v[0][0::2]
    beta  = v[0][1::2]

    amp = np.sqrt( alpha**2 + beta**2 )
    phase = np.arctan(- beta/alpha)

    h = np.zeros(len(t))
    for ik in range(nK):
        h +=  np.sqrt(alpha[ik]**2 + beta[ik]**2) * np.cos( 2.*np.pi*ik * (f0/fs) * np.arange(0, len(t), 1 )  + phase[ik] )

    #plt.matshow(A, aspect='auto')
    #plt.colorbar()

    #plt.figure()
    #plt.plot(alpha)
    #plt.plot(beta)
    #plt.plot(amp)

    #plt.figure()
    #plt.plot(h)
    #plt.title("modelled noise")

    return h

if __name__ == "__main__":
    import matplotlib.pyplot as plt 

    f0 = 60      # Hz
    delta = np.random.rand()
    fs = 50000  #1e4    
    t = np.arange(0, 1, 1/fs)
    phi = .234
    A = 1.0
    nK = 20
    sN = A * np.sin( (delta+f0)*2*np.pi*t + phi ) + np.random.normal(0,.1,len(t)) 
    sNc = A * np.sin( (delta+f0)*2*np.pi*t + phi ) 
    h = harmonic(sN, f0, fs, nK, t)

    plt.figure()
    plt.plot(t, sN, label="sN")
    plt.plot(t, sN-h, label="sN-h")
    plt.plot(t, h, label='h')
    plt.title("true noise")
    plt.legend()

    plt.figure()
    plt.plot(t, sN-sNc, label='true noise')
    plt.plot(t, sN-h, label='harmonic removal')
    plt.legend()
    plt.title("true noise")
    
    plt.show()

    print("hello")
