# #################################################################################
# # GJI final pub specs                                                           #
# import matplotlib                                                               #
# from matplotlib import rc                                                       #
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{timet,amsmath}"]      #
# rc('font',**{'family':'serif','serif':['timet']})                               #
# rc('font',**{'size':8})                                                         #
# rc('text', usetex=True)                                                         #
# # converts pc that GJI is defined in to inches                                  # 
# # In GJI \textwidth = 42pc                                                      #
# #        \columnwidth = 20pc                                                    #
# def pc2in(pc):                                                                  #
#     return pc*12/72.27                                                          #
# #################################################################################

#from GJIPlot import *

import numpy as np
import matplotlib.pyplot as plt
#from invertColours import *
from akvo.tressel.decay import *
from scipy import signal

def quadrature(T, vL, wL, dt, xn, DT, t):
        # decimate
    # blind decimation
    # 1 instead of T 
    irsamp = int(T) * int(  (1./vL) / dt) # real 
    iisamp =       int(  ((1./vL)/ dt) * ( .5*np.pi / (2.*np.pi) ) ) # imaginary
   

    dsamp =  int( DT / dt) # real 

    iisamp += dsamp

    ############################################################
    # simple quadrature-detection via sampling
    xr = xn[dsamp::irsamp]
    xi = xn[iisamp::irsamp]
    phase = np.angle( xr + 1j*xi )
    abse = np.abs( xr + 1j*xi )

    # times
    #ta = np.arange(0, TT, dt)
    #te = np.arange(DT, TT, TT/len(abse))

    #############################################################
    # hilbert transform
    ht = signal.hilbert(xn) #, 100))
    he = np.abs(ht)         #, 100))
    hp = ((np.angle(ht[dsamp::irsamp]))) 

    #############################################################
    # Resample ht
    #htd = signal.decimate(he, 100, ftype='fir') 
    #td = signal.decimate(t, 100, ftype='fir')
    #[htd, td] = signal.resample(he, 21, t) 
    #toss first, and use every third 
    #htd = htd[1::3]
    #td = td[1::3]

    #############################################################
    # Pre-envelope
    #gplus = xn + 1j*ht  

    #############################################################
    # Complex envelope
    #gc = gplus / np.exp(1j*wL*t)    

    #############################################################
    ## Design a low-pass filter
    FS = 1./dt                                           # sampling rate
    FC = 10.05/(0.5*FS)                                 # cutoff frequency at 0.05 Hz
    N = 11                                               # number of filter taps
    a = [1]                                              # filter denominator
    b = signal.firwin(N, cutoff=FC, window='hamming')    # filter numerator

    #############################################################
    ## In-phase 
    #2*np.cos(wL*t)  
    dw = 0 # -2.*np.pi*2
    Q = signal.filtfilt(b, a, xn*2*np.cos((wL+dw)*t))  # X
    I = signal.filtfilt(b, a, xn*2*np.sin((wL+dw)*t))  # Y

    ###############################################
    # Plots
    #plt.plot(ht.real)
    #plt.plot(ht.imag)
    #plt.plot(np.abs(ht))
    #plt.plot(gc.real)
    #plt.plot(gc.imag)

    #plt.plot(xn)
    #plt.plot(xn)
    #plt.plot(ta, xn)
    #plt.plot(te, abse, '-.', linewidth=2, markersize=10)
    #plt.plot(ta, he, '.', markersize=10 )
    #plt.plot(td, htd, color='green', linewidth=2)
    # Phase Plots
    #ax2 = plt.twinx()
    #ax2.plot(te, hp, '.', markersize=10, color='green' )
    #ax2.plot(te, phase, '-.', linewidth=2, markersize=10, color='green')


    return Q[N:-N], I[N:-N], t[N:-N]

#     #####################################################################
#     # regress raw signal
#     
#     #[peaks, times, ind] = peakPicker(xn, wL, dt)
#     #[a0,b0,rt20] =  regressCurve(peaks,  times) #,sigma2=1,intercept=True):
#     
#     dsamp = int( DT / dt) # real 
#     # regress analytic signal
#     [a0,b0,rt20] =  regressCurve(he[dsamp::],  t[dsamp::], intercept=True) #,sigma2=1,intercept=True):
#     #[b0,rt20] =  regressCurve(he[dsamp::],  t[dsamp::], intercept=False) #,sigma2=1,intercept=True):
#     #[a0,b0,rt20] =  regressCurve(he,  t) #,sigma2=1,intercept=True):
#    
#     # regress downsampled 
#     [a,b,rt2] =  regressCurve(abse,  t[dsamp::irsamp], intercept=True) #,sigma2=1,intercept=True):
#     #[b,rt2] =  regressCurve(htd,  td, intercept=False) #,sigma2=1,intercept=True):
#     
#     return irsamp, iisamp, htd, b0, rt20, ta, b, rt2, phase, td, he, dsamp
#     #return irsamp, iisamp, abse, a0, b0, rt20, times, a, b, rt2, phase

def RotateAmplitude(X, Y, zeta, df, t):
    V = X + 1j*Y
    return np.abs(V) * np.exp( 1j * ( np.angle(V) - zeta - 2.*np.pi*df*t ) )
    #return np.abs(V) * np.exp( 1j * ( np.angle(V) - zeta - df*t ) )

def gateIntegrate(T2D, T2T, gpd, sigma, stackEfficiency=2.):
    """ Gate integrate the signal to gpd, gates per decade
        T2D = the time series to gate integrate, complex 
        T2T = the abscissa values 
        gpd = gates per decade 
        sigma = estimate of standard deviation for theoretical gate noise 
        stackEfficiency = exponential in theoretical gate noise, 2 represents ideal stacking
    """
    
    # use artificial time gates so that early times are fully captured
    T2T0 = T2T[0]
    T2TD = T2T[0] - (T2T[1]-T2T[0])
    T2T -= T2TD
    
    #####################################
    # calculate total number of decades #
    # windows edges are approximate until binning but will be adjusted to reflect data timing, this 
    # primarily impacts bins with a few samples  
    nd = np.log10(T2T[-1]/T2T[0])               
    tdd = np.logspace( np.log10(T2T[0]), np.log10(T2T[-1]), (int)(gpd*nd)+1, base=10, endpoint=True) 
    tdl = tdd[0:-1]                 # approximate window left edges
    tdr = tdd[1::]                  # approximate window right edges
    td = (tdl+tdr) / 2.             # approximate window centres


    Vars = np.zeros( len(td) ) 
    htd = np.zeros( len(td), dtype=complex )
    isum = np.zeros( len(td), dtype=int )  

    ii = 0
    for itd in range(len(T2T)):
        if ( round(T2T[itd], 4) > round(tdr[ii], 4) ):
            ii += 1
            # correct window edges to centre about data 
            tdr[ii-1] = (T2T[itd-1]+T2T[itd])*.5 
            tdl[ii  ] = (T2T[itd-1]+T2T[itd])*.5
        isum[ii] += 1
        htd[ii] += T2D[ itd ]
        Vars[ii] += sigma**2
        
    td = (tdl+tdr) / 2.             # actual window centres
    sigma2 = np.sqrt( Vars * ((1/(isum))**stackEfficiency) ) 

    # Reset abscissa where isum == 1 
    # when there is no windowing going on 
    td[isum==1] = T2T[0:len(td)][isum==1]

    tdd = np.append(tdl, tdr[-1])

    htd /= isum # average
    T2T += T2TD 
    return td+T2TD, htd, tdd+T2TD, sigma2, isum  # centre abscissa, data, window edges, error 

if __name__ == "__main__":

    dt = 1e-4
    TT = 1.5
    t = np.arange(0, TT, dt)
    vL = 2057.
    wL =  2.*np.pi*vL
    wL2 = 2.*np.pi*(vL-2.5) #-2) #-2.2) # 3 Hz off 
    zeta = -np.pi/6. #4.234
    t2 = .150

    xs = np.exp(-t/t2) * np.cos(wL2*t + zeta) 
    xe = np.exp(-t/t2)    
    xn = xs + np.random.normal(0,.1,len(xs))# + (np.sign(xs) 
            #    np.random.random_integers(-1,1,len(xs))*0.6*np.random.lognormal(0, .35, len(xs)) + \
            #    np.random.random_integers(-1,1,len(xs))*.004*np.random.weibull(.25, len(xs)), 60)))

    # quadrature detection downsampling
    T = 50    # sampling period, grab every T'th oscilation
    DT = .002 #85  # dead time ms
    #[irsamp, iisamp, abse, b0, rt20, times, b, rt2, phase, tdec, he, dsamp] = quadDetect(T, vL, wL, dt, xn, DT)
    
    [Q, I, tt] = quadrature(T, vL, wL, dt, xn, DT, t)
    [E0,df,phi,T2] = quadratureDetect(Q, I, tt)
    print("df", df)
    D = RotateAmplitude(I, Q, phi, df, tt)

    fig = plt.figure(figsize=[pc2in(20), pc2in(14)]) #
    ax1 = fig.add_axes([.125,.2,.8,.7])
    #ax1.plot(tt*1e3, np.exp(-tt/t2), linewidth=2, color='black', label="actual")   
    ax1.plot(tt*1e3, D.imag, label="CA", color='red')    
    ax1.plot(t*1e3, xn, color='blue', alpha=.25)
    ax1.plot(tt*1e3, I, label="inphase", color='blue')
    ax1.plot(tt*1e3, Q, label="quadrature", color='green')
    
    #ax1.plot(tt*1e3, np.angle( Q + 1j*I), label="angle", color='purple')


    GT, GD = gateIntegrate( D.imag, tt, 10  )
    GT, GDR = gateIntegrate( D.real, tt, 10  )
    GT, GQ = gateIntegrate( Q, tt, 10  )
    GT, GI = gateIntegrate( I, tt, 10  )
    #ax1.plot(tt*1e3, np.arctan( Q/I), label="angle", color='purple')
    #ax1.plot(GT*1e3, np.real(GD), 'o', label="GATE", color='purple')
    #ax1.plot(GT*1e3, np.real(GDR), 'o', label="GATE Real", color='red')
    #ax1.plot(GT*1e3, np.arctan( np.real(GQ)/np.real(GI)), 'o',label="GATE ANGLE", color='magenta')
    

    ax1.set_xlabel(r"time [ms]") 
    ax1.set_ylim( [-1.25,1.65] )
    
    #light_grey = np.array([float(248)/float(255)]*3)
    legend = plt.legend( frameon=True, scatterpoints=1, numpoints=1, labelspacing=0.2  )
    #rect = legend.get_frame()
    fixLeg(legend)
    #rect.set_color('None')
    #rect.set_facecolor(light_grey)
    #rect.set_linewidth(0.0)
    #rect.set_alpha(0.5)

    # Remove top and right axes lines ("spines")
    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax1.spines[spine].set_visible(False)
    #ax1.xaxis.set_ticks_position('none')
    #ax1.yaxis.set_ticks_position('none')
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    plt.savefig('rotatetime.pdf',dpi=600)
    plt.savefig('rotatetime.eps',dpi=600)

    # phase part
    plt.figure()
    plt.plot( tt*1e3, D.real, label="CA", color='red' )

    plt.show()
    exit()
