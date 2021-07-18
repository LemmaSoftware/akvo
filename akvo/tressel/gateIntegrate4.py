from __future__ import division 

import matplotlib as mpl
mpl.use('pdf')

#from rasterize import rasterize_and_save

import matplotlib.patches as mpatches
from pwctime import pwcTime
from logbarrier import * 
from perlin import perlin
from scipy import stats
import cmocean
import sys 
import numpy as np 
import seaborn as sns

def bootstrapWindows(N, nboot, isum, adapt=False):
    """ Bootstraps noise as a function of gate width
        N = input noise signal 
        nboot = number of boostrap windows to perform 
        isum = length of windows (L_i)
        adapt = reduce nboot as window size increases
    """
    nc = np.shape(N)[0]
    Means = {}

    if adapt:
        Means = -9999*np.ones((len(isum), nboot//isum[0])) # dummy value
        for ii, nwin in enumerate(isum):  
            for iboot in range(nboot//isum[ii]):
                cs = np.random.randint(0,nc-nwin)
                Means[ii,iboot] = np.mean( N[cs:cs+nwin] )
        Means = np.ma.masked_less(Means, -9995)

    else:
        Means = np.zeros((len(isum), nboot))
        for ii, nwin in enumerate(isum):  
            for iboot in range(nboot):
                cs = np.random.randint(0,nc-nwin)
                Means[ii,iboot] = np.mean( N[cs:cs+nwin] )

    return Means, np.array(isum)

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
        if ( T2T[itd] > tdr[ii] ):
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
    T2T += T2TD # not used  
    return td+T2TD, htd, tdd+T2TD, sigma2, isum  # centre abscissa, data, window edges, error 

PhiD = []
def invert(Time, t, v, sig, lambdastar):
    """ helper function that simply calls logBarrier, here to allow for drop in repacement  
    """
    #model = logBarrier(Time.Genv, 1e-2*v, Time.T2Bins, MAXITER=5000, sigma=1e-2*sig, alpha=1e6, smooth="Both") 
    model = logBarrier(Time.Genv, 1e-2*v, Time.T2Bins, lambdastar, MAXITER=750, sigma=1e-2*sig, alpha=1e6, smooth="Smallest") 
    PhiD.append(model[2])
    return model

def gateTest(vc, vgc, pperlin, boot, lamdastar):
    """ Performs gate integration and adds random noise 
        vc = clean data (dense)
        vgc = clean data at gates 
        boot = if "boot" then bootstrap the gate noise 
        lambdastar = l-curve or discrepency principle
        pperlin = percent perlin noise, noise floor is maintained at 2.00 PU 
    """

    t = np.arange(2e-4, .3601, 2e-4)
    zeta = np.pi / 3.
    
    v = np.copy(vc) # important!  

    # Scaling factors to keep noise floor constant with increasing levels of 
    # Perlin noise. These were determined using populations of 5,000 and hold to 
    # two significant digits (i.e, 2.00 PU) 
    PF = {0.0:0,\
          2.5:.450,\
          5.0:.6125,\
          7.5:.765,\
          10.0:.87375,\
          12.5:.9725,\
          15.0:1.05,\
          17.5:1.1275,\
          20.0:1.20,\
          22.5:1.265,\
          25.0:1.325}
 
    # random noise
    np.random.seed() # necessary for thread pool, otherwise all threads can get same numbers 
    sigma = 2.*(1.-1e-2*pperlin)
    eps = np.random.normal(0, sigma, len(t)) + \
       1j*np.random.normal(0, sigma, len(t)) 
    eps += PF[pperlin] * perlin(len(t), L=.3601, sigma_f=.005, sigma_r=0.72) # 1 PU std
    v += eps

    # Noise based on residual  
    sigmahat = np.std( v.imag )
    
    gt, gd, we, err, isum = gateIntegrate(v.real, t, 20, sigmahat)
    ge = np.copy(err)
    if boot=="boot":
        Means, isum2 = bootstrapWindows(v.imag, 20000, isum[isum!=1], adapt=True)
        # STD 
        #err[isum!=1] = np.ma.std(Means, axis=1, ddof=1)[isum!=1]
        # MAD, only for windows > 1 
        c = stats.norm.ppf(3./4.)
        err[isum!=1] = np.ma.median(np.ma.abs(Means), axis=1) / c 
    if boot=="uniform":
        err = sigmahat
 
    return gt, gd, gd-vgc, err, v.real, t, isum

if __name__ == "__main__":

    import multiprocessing 
    import itertools 

    from GJIPlot import *
    import numpy as np 
    import matplotlib.pyplot as plt 
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib import ticker
    from collections import OrderedDict

    if len(sys.argv)<4:
        print ( "Python script for generating plots used in GJI publication")
        print ( "useage:")
        print ( "python gateIntegrate4.py   NoiseType   Sigma_i  Lambda*   " )
        exit()

    if sys.argv[1] not in ['0.0','2.5','5.0','7.5','10.0','12.5','15.0','17.5','20.0','22.5','25.0']: 
        print ( "PercentPerlin: [0.0,2.5,5.0...25.0] ", "got", sys.argv[1])
        exit(1)

    if sys.argv[2] != "gauss" and sys.argv[2] != "boot" and sys.argv[2] != "uniform":
        print ( "Sigma_i: gauss | boot | uniform")
        exit(1)
    
    if sys.argv[3] != "lcurve" and sys.argv[3] != "discrepency": 
        print ( "Lambda*: lcurve | discrepency ")
        exit(1)

    #offwhite = (.98,.98,.98)
    offwhite = (1.,1.,1.) 
    mDarkBrown = '#eb811b' # alert colour 
    mDarkTeal = '#23373b'
    mLightBrown= "#EB811B"
    mLightGreen = "#14B03D"

    # Time series plot
    fig = plt.figure(figsize=(pc2in(18),pc2in(18)), facecolor=offwhite)
    ax2  = fig.add_axes([.195, .175, .750, .75], facecolor=offwhite)  # time
    
    # Main plot 
    fig2 = plt.figure(figsize=(pc2in(20),pc2in(2*.5*20)))
    ax1  = fig2.add_axes([.175, .410*1.5, .6,   .225*1.5])
    ax1c = fig2.add_axes([.800, .410*1.5, .025, .225*1.5])
    ax3  = fig2.add_axes([.175, .100*1.5, .495, .225*1.5], facecolor='None')
    ax3r  = fig2.add_axes([.175, .100*1.5, .495, .225*1.5], facecolor='None', rasterized=True, sharex=ax3, sharey=ax3)
    ax3b = fig2.add_axes([.825, .100*1.5, .1,   .225*1.5])

    SIG = []
    ER = []
    GD = []
    GT = []
    V = []
    MOD = []
    CONV = []
    PHID = []
    PHIM = []
    LSTAR = []
    ns = 10000 #10000  #10000 # number of realizations for PDF 
    ni = 5000  #5000  #1000  # number of inversions to plot 
    t = np.arange(2e-4, .3601, 2e-4) # CMR sampling
 
    #CMAP = cmocean.cm.solar
    CMAP = cmocean.cm.gray_r
    #CMAP = cmocean.cm.haline
    #CMAP = cmocean.cm.tempo

    ##############################################
    # set up model 
    lowT2 = .001
    hiT2 = 1.0 
    nT2 = 30
    spacing = "Log_10"
    Time = pwcTime()
    Time.setT2(lowT2, hiT2, nT2, spacing) 
    Time.setSampling( np.arange(2e-4, .3601, 2e-4) )
    Time.generateGenv()
    tmod = np.zeros(nT2)
    tmod [8] = .15  # distribution centres...to be smoothed
    tmod [20] = .1
    for i in range(2):
        tmod = np.convolve(tmod, np.array([.0625,.125,.1875,.25,.1875,.125,.0625]), 'same') 

    vc = 100. * np.dot(Time.Genv, tmod) + 0j # in PU
    gt, gd, we, err, isum = gateIntegrate(vc, t, 20, 3)

    ##############################################
    # Set up inversion 
    Time = pwcTime()
    Time.setT2(lowT2, hiT2, nT2, spacing) 
    Time.setSampling( gt ) 
    Time.generateGenv()
    vgc = 100.*np.dot(Time.Genv, tmod) + 0j # in PU

    # make the Pool of workers
    print("pool gate integrate")
    with multiprocessing.Pool() as pool: 
        results = pool.starmap(gateTest, zip(np.tile(vc, (ns, 1)), np.tile(vgc, (ns,1)), itertools.repeat(eval(sys.argv[1])), \
                                                                                         itertools.repeat(sys.argv[2]), \
                                                                                         itertools.repeat(sys.argv[3])))
    print("done pool gate integrate")
   
    # parse out results 
    for i in range(ns):
        gt,gd,ge,err,v,vt,isum = results[i] 
        V.append(v.real)
        GT.append(gt.real)
        GD.append(gd.real)
        ER.append( ge.real / err.real )
        SIG.append( err.real )

    print("pool inversions")
    with multiprocessing.Pool() as pool: 
        invresults = pool.starmap(invert, zip(itertools.repeat(Time), GT[0:ni], GD[0:ni], SIG[0:ni], itertools.repeat(sys.argv[3]) )) 
    #print("done pool inversions",results[:][0])

    # Parse results 
    for i in range(ns):
        #print("Sym %", round(100.*i/(float)(1)/(float)(ns)))
        # invert
        if i < ni:
            #mod, conv, phid = invert(Time, gt, gd.real, err)
            if sys.argv[3] == "discrepency":
                mod, conv, phid_final = invresults[i] 
            else:
                mod, conv, phid_final, phim, phid, lstar = invresults[i] 
            MOD.append(mod)
            CONV.append(conv)
            PHID.append(phid)
            PHIM.append(phim)
            LSTAR.append(lstar)

    PHIM = np.array(PHIM)
    PHID = np.array(PHID)
    ER = np.array(ER)
    MOD = np.array(MOD)
    GD = np.array(GD)

    ####################
    # Time series plot #       
    ax2.plot( 1e3*vt, V[0], color=mDarkTeal, label="$V_N$", linewidth=1, zorder=-32) #, rasterized=True) 
    ax2.errorbar( 1e3*gt, GD[0], yerr=SIG[0], fmt='.', markersize=6, color=mLightBrown, label="$V_G$")
    ax2.set_ylim([-10,30])
    leg1 = ax2.legend( labelspacing=0.2, scatterpoints=1, numpoints=1, frameon=True )   
    fixLeg(leg1)
    ax2.set_xscale("log", nonposx='clip')  
    ax2.set_ylabel(r"$V_N$ (PU)")
    ax2.get_xaxis().set_major_formatter(FormatStrFormatter('%1.0f'))
    ax2.set_xlabel("time (ms)")
    deSpine(ax2)
    fig.savefig( sys.argv[1] + "-" + sys.argv[2] + "-" + sys.argv[3] + "-ts.pdf", dpi=400, facecolor=offwhite,edgecolor=offwhite)

    # histogram of error statistic
    bins = np.linspace( -3, 3, 40, endpoint=True )
    HIST = []
    for i in range(0,np.shape(ER)[1]):
        hist, edges = np.histogram(ER[:,i], bins=bins, density=False)        
        HIST.append(hist)
    HIST =  np.array(HIST)/(float)(ns) # normalize 
        
    im = ax1.pcolor(1e3*we, edges, HIST.T, cmap=CMAP, vmin=0, vmax=.1, rasterized=True)
    im.set_edgecolor('face')
    cb = plt.colorbar(im, ax1c, label=r"probability density", format=FormatStrFormatter('%1.2f'))
    cb.solids.set_rasterized(True) 
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb.locator = tick_locator
    cb.update_ticks()

    ax1.set_xscale("log", nonposx='clip')
    ax1.get_xaxis().set_major_formatter(FormatStrFormatter('%1.0f'))
    ax1.set_xlabel("time (ms)")
    ax1.set_ylabel(r"gate error $\left( \left( {V_G - V_T} \right) / {\tilde{\sigma_i}} \right)$")

    LMT2 = [] 
    THETA = []
    MODERR = []
       
    # plot a random sample of ns instead?  
    for i in range(ni):
        # plot log mean and amplitude
        model = MOD[i] 
        theta = np.sum( model )
        LogMeanT2 = np.exp(np.sum( model * np.log( Time.T2Bins ) ) / theta ) 
        LMT2.append(LogMeanT2)
        THETA.append( np.sum(model)  )
        MODERR.append( np.linalg.norm(model-tmod) )
                
    CONV = np.array(CONV)
    THETA = np.array(THETA)
    MOD = np.array(MOD)
    MODERR = np.array(MODERR)
        
    #############################
    # plot all models, 1 colour #
    ires = ax3r.plot( 1e3*np.tile(Time.T2Bins, (np.sum(np.array(CONV)) ,1)).T , 1e2*MOD.T, color=mDarkTeal, alpha=.01, lw=.5, label="$\mathbf{f}_I$", zorder=0, rasterized=True) 
    lns2, = ax3r.plot(1e3*Time.T2Bins, 1e2*tmod, color=mLightBrown, linewidth=2, label="$\mathbf{f}_T$")

    handles, labels = ax3r.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    leg3 = ax3r.legend(by_label.values(), by_label.keys(), labelspacing=0.2, scatterpoints=1, numpoints=1, frameon=True , loc="upper right")            
    for line in leg3.get_lines():
        line.set_linewidth(1)
    for lh in leg3.legendHandles: 
        lh.set_alpha(1)
    fixLeg(leg3)

    ###########################
    # Error histogram on side #
    ax3b.hist( 1e2*MODERR, bins='auto', orientation="horizontal", color=mDarkTeal, stacked=True, density=True, range=(0,20)) 
    ax3b.axhline(1e2*np.mean(MODERR), linewidth=1.25, color=mLightBrown) #, color=CMAP(0.7), zorder=1)
    deSpine(ax3b)
    ax3b.set_xscale("log", nonposx='clip')  
    ax3b.set_ylabel(r"$\Vert \mathbf{f}_I -\mathbf{f}_T \Vert$") # %(m$^3$/m$^3$)") #, color="C0")   
    ax3b.set_xlabel("log probability\ndensity") #, color="C0")   
    ax3.set_xlim( (1e3*Time.T2Bins[0], 1e3*Time.T2Bins[-1]) ) 
    ax3.set_ylim( (0,5) )  
    ax3.set_xlabel("$T_2$ (ms)")   
    ax3.set_ylabel("partial water content (PU)") #, color="C0")   
    ax3.set_xscale("log", nonposx='clip')  
    ax3.get_xaxis().set_major_formatter(FormatStrFormatter('%1.0f'))
    plt.setp(ax3r.get_xticklabels(), visible=False)
    plt.setp(ax3r.get_yticklabels(), visible=False)
    deSpine(ax3)
    deSpine(ax3r)
    
    np.save("pperlin" + str(round(1e1*eval(sys.argv[1]))) + "-" + sys.argv[2] + "-" + sys.argv[3] + "-err", MODERR) 

    plt.savefig("pperlin" + str(round(1e1*eval(sys.argv[1]))) + "-" + sys.argv[2] + "-" + sys.argv[3] + ".pdf", dpi=600, facecolor=offwhite, edgecolor=offwhite)
    
    plt.show()

