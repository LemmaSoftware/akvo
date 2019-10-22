from akvo.tressel.SlidesPlot import *
import numpy as np
import sys
import matplotlib.pyplot as plt
import cmocean
from pylab import meshgrid 
from akvo.tressel.logbarrier import *
import yaml,os

from matplotlib.colors import LogNorm
from matplotlib.colors import LightSource
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FormatStrFormatter
import cmocean
from akvo.tressel.lemma_yaml import * 

def buildKQT(K0,tg,T2Bins):
    """ 
        Constructs a QT inversion kernel from an initial amplitude one.  
    """
    nlay, nq = np.shape(K0)
    nt2 = len(T2Bins)
    nt = len(tg)
    KQT = np.zeros( ( nq*nt,nt2*nlay) )
    for iq in range(nq):
        for it in range(nt):
            for ilay in range(nlay):
                for it2 in range(nt2):
                    #KQT[iq*nt + it,ilay*nt2+it2] = K0[ilay,iq]*np.exp(-((10+tg[it])*1e-3)/(1e-3*T2Bins[it2]))
                    KQT[iq*nt + it,ilay*nt2+it2] = K0[ilay,iq]*np.exp(-((10+tg[it])*1e-3)/(1e-3*T2Bins[it2]))
    return KQT

def loadAkvoData(fnamein, chan):
    """ Loads data from an Akvo YAML file. The 0.02 is hard coded as the pulse length. This needs to be 
        corrected in future kernel calculations. The current was reported but not the pulse length. 
    """
    fname = (os.path.splitext(fnamein)[0])
    with open(fnamein, 'r') as stream:
        try:
            AKVO = (yaml.load(stream, Loader=yaml.Loader))
        except yaml.YAMLError as exc:
            print(exc)
            exit()

    Z = np.zeros( (AKVO.nPulseMoments, AKVO.Gated["Pulse 1"]["abscissa"].size ) )
    ZS = np.zeros( (AKVO.nPulseMoments, AKVO.Gated["Pulse 1"]["abscissa"].size ) )
    for q in range(AKVO.nPulseMoments):
        Z[q] = AKVO.Gated["Pulse 1"][chan]["Q-"+str(q) + " CA"].data
        if chan == "Chan. 1":
            ZS[q] = AKVO.Gated["Pulse 1"][chan]["STD"].data
        elif chan == "Chan. 2":
            ZS[q] = AKVO.Gated["Pulse 1"][chan]["STD"].data
        elif chan == "Chan. 3":
            ZS[q] = AKVO.Gated["Pulse 1"][chan]["STD"].data
        elif chan == "Chan. 4":
            ZS[q] = AKVO.Gated["Pulse 1"][chan]["STD"].data
        else:
            print("DOOM!!!")
            exit()
    #Z *= 1e-9 
    #ZS *= 1e-9 

    J = AKVO.Pulses["Pulse 1"]["current"].data 
    J = np.append(J,J[-1]+(J[-1]-J[-2]))
    Q = AKVO.pulseLength[0]*J
    return Z, ZS, AKVO.Gated["Pulse 1"]["abscissa"].data  #, Q

def catLayers(K0):
    K = np.zeros( (len(K0.keys()), len(K0["layer-0"].data)) , dtype=complex )
    for lay in range(len(K0.keys())):
        #print(K0["layer-"+str(lay)].data) #    print (lay)
        K[lay] =K0["layer-"+str(lay)].data #    print (lay)
    return 1e9*K                           # invert in nV

def loadK0(fname):
    """ Loads in initial amplitude kernel
    """
    print("loading K0", fname)
    with open(fname) as f:
        K0 = yaml.load(f, Loader=yaml.Loader)
    K = catLayers(K0.K0)
    ifaces = np.array(K0.Interfaces.data)
    return ifaces, np.abs(K)



def main():

    if (len (sys.argv) < 3):
        print ("python invertTA.py  K0.dat  Data.yaml")
    print (sys.argv[1])
    with open(sys.argv[1], 'r') as stream:
        try:
            cont = (yaml.load(stream, Loader=yaml.Loader))
        except yaml.YAMLError as exc:
            print(exc)
    ###############################################
    # Load in data
    ###############################################
    V = []
    VS = []
    tg = 0
    for dat in cont['data']:
        for ch in cont['data'][dat]['channels']:
            print("dat", dat, "ch", ch)
            v,vs,tg = loadAkvoData(dat, ch)
            V.append(v)
            VS.append(vs)
    for iv in range(1, len(V)):
        V[0] = np.concatenate( (V[0], V[iv]) )
        VS[0] = np.concatenate( (VS[0], VS[iv]) )
    V = V[0]
    VS = VS[0]
    
    #plt.matshow(V, cmap='RdBu', vmin=-np.max(np.abs(V)), vmax=np.max(np.abs(V)))
    #plt.title("data")
    #plt.colorbar()
    #plt.show()
    #exit()    

    #plt.matshow(VS, cmap='inferno', vmin=-np.max(np.abs(VS)), vmax=np.max(np.abs(VS)))
    #plt.title("error")
    #plt.colorbar()

    ###############################################
    # Load in kernels
    ###############################################
    K0 = []
    for kern in cont["K0"]:
        ifaces,k0 = loadK0( kern )
        K0.append(k0)
    for ik in range(1, len(K0)):
        K0[0] = np.concatenate( (K0[0].T, K0[ik].T) ).T
    K0 = K0[0]
    #plt.matshow(K0)
    
    ###############################################
    # Build full kernel
    ############################################### 
    T2Bins = np.logspace( np.log10(cont["T2Bins"]["low"]), np.log10(cont["T2Bins"]["high"]), cont["T2Bins"]["number"], endpoint=True, base=10)  
    KQT = buildKQT(K0,tg,T2Bins)
    #plt.matshow(KQT)

    # Chan. 1    reg = 1.5   
    # Chan. 2    reg = 1.35
    # Chan. 4    reg = 1.95  
 
    ###############################################
    # Invert
    ############################################### 
    print("Calling inversion", flush=True)
    inv, ibreak, errn, phim, phid, mkappa = logBarrier(KQT, np.ravel(V), T2Bins, "lcurve", MAXITER=150, sigma=np.ravel(VS), alpha=1e6, smooth="Smallest" ) #, smooth=True) # 
                     #logBarrier(A, b, T2Bins, lambdastar, x_0=0, xr=0, alpha=10, mu1=10, mu2=10, smooth=False, MAXITER=70, fignum=1000, sigma=1, callback=None):
    #return x, ibreak, np.sqrt(phid/len(b)), PHIM, PHID/len(b), np.argmax(kappa)


    ###############################################
    # Appraise
    ############################################### 
    pre = np.dot(KQT,inv) 
    PRE = np.reshape( pre, np.shape(V)  )
    plt.matshow(PRE, cmap='Blues')
    plt.gca().set_title("predicted")
    plt.colorbar()

    DIFF = (PRE-V) / VS
    md = np.max(np.abs(DIFF))
    plt.matshow(DIFF, cmap=cmocean.cm.balance, vmin=-md, vmax=md)
    plt.gca().set_title("misfit / $\widehat{\sigma}$")
    plt.colorbar()
    
    plt.matshow(V, cmap='Blues')
    plt.gca().set_title("observed")
    plt.colorbar()

    T2Bins = np.append( T2Bins, T2Bins[-1] + (T2Bins[-1]-T2Bins[-2]) )
    
    print( np.shape(inv), len(ifaces)-1,cont["T2Bins"]["number"] )
    INV = np.reshape(inv, (len(ifaces)-1,cont["T2Bins"]["number"]) )
    Y,X = meshgrid( ifaces, T2Bins )
    fig = plt.figure( figsize=(pc2in(17.0),pc2in(18.)) )
    ax1 = fig.add_axes( [.2,.15,.6,.7] )
    im = ax1.pcolor(X, Y, INV.T, cmap=cmocean.cm.tempo) #cmap='viridis')
    im.set_edgecolor('face')
    ax1.set_xlim( T2Bins[0], T2Bins[-1] )
    ax1.set_ylim( ifaces[-1], ifaces[0] )
    cb = plt.colorbar(im, label = u"PWC (m$^3$/m$^3$)") #, format='%1.1f')
    cb.locator = MaxNLocator( nbins = 4)
    cb.ax.yaxis.set_offset_position('left')                         
    cb.update_ticks()
 
    #ax1.set_xlabel(u"$T_2^*$ (ms)")
    #ax1.set_ylabel(u"depth (m)$")
    
    ax1.get_xaxis().set_major_formatter(FormatStrFormatter('%1.0f'))
    ax1.get_yaxis().set_major_formatter(FormatStrFormatter('%1.0f'))
    ax1.xaxis.set_major_locator( MaxNLocator(nbins = 4) )   

    #ax1.xaxis.set_label_position('top') 

    ax2 = ax1.twiny()
    ax2.plot( np.sum(INV, axis=1), (ifaces[1:]+ifaces[0:-1])/2 ,  color='red' )
    ax2.set_xlabel(u"total water (m$^3$/m$^3$)")
    ax2.set_ylim( ifaces[-1], ifaces[0] )
    ax2.xaxis.set_major_locator( MaxNLocator(nbins = 3) )   
    ax2.get_xaxis().set_major_formatter(FormatStrFormatter('%0.2f'))
    #ax2.xaxis.set_label_position('bottom') 

    plt.savefig("akvoInversion.pdf")
    
    plt.show()

if __name__ == "__main__":
    main() 

