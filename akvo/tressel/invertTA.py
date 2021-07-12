from akvo.tressel.SlidesPlot import *
import numpy as np
import sys
import matplotlib.pyplot as plt
import cmocean
from pylab import meshgrid 
from akvo.tressel.logbarrier import *
import yaml,os

import multiprocessing 
import itertools
 
from scipy.linalg import svd 

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.colors import LightSource
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import Normalize

import cmocean
from akvo.tressel.lemma_yaml import *
from akvo.tressel import nonlinearinv as nl 

import pandas as pd


import matplotlib.colors as colors

# From https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def buildKQT(K0,tg,T2Bins):
    """ 
        Constructs a QT inversion kernel from an initial amplitude one.  
    """
    nlay, nq = np.shape(K0)
    nt2 = len(T2Bins)
    nt = len(tg)
    KQT = np.zeros( ( nq*nt,nt2*nlay), dtype=np.complex128 )
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
    return Z, ZS, AKVO.Gated["Pulse 1"]["abscissa"].data, Q

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
    return ifaces, K
    #return ifaces, np.abs(K)

def invertDelta(G, V_n, T2Bins, sig, alphastar):
    """ helper function that simply calls logBarrier, simplfies parallel execution
    """
    model = logBarrier(G, V_n, T2Bins, "Single", MAXITER=1, sigma=sig, alpha=alphastar, smooth="Smallest") 
    return model


def main():

    if (len (sys.argv) < 2):
        print ("akvoQT   invertParameters.yaml")
        exit()
    
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
    QQ = []
    tg = 0
    for dat in cont['data']:
        for ch in cont['data'][dat]['channels']:
            print("dat", dat, "ch", ch)
            v,vs,tg,Q = loadAkvoData(dat, ch)
            V.append(v)
            VS.append(vs)
            QQ.append(Q)
    for iv in range(1, len(V)):
        V[0] = np.concatenate( (V[0], V[iv]) )
        VS[0] = np.concatenate( (VS[0], VS[iv]) )
    V = V[0]
    VS = VS[0]

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

    #np.save("ifaces", ifaces)
    #exit()

    #plt.matshow(np.real(K0))
    #plt.show()
    #exit()

    ##############################################################    
    # VERY Simple Sensitivity based calc. of noise per layer     #
    # minimally useful, but retained for backwards compatibility #
    maxq = np.argmax(np.abs(K0), axis=1)
    maxK = .1 *  np.abs(K0)[ np.arange(0,len(ifaces)-1), maxq ] # 10% water is arbitrary  
    SNR = maxK / (VS[0][0])

    ###############################################
    # Build full kernel
    ############################################### 

    T2Bins = np.logspace( np.log10(cont["T2Bins"]["low"]), np.log10(cont["T2Bins"]["high"]), cont["T2Bins"]["number"], endpoint=True, base=10) 
    T2Bins2 = np.append( T2Bins, T2Bins[-1] + (T2Bins[-1]-T2Bins[-2]) )
    NT2 = len(T2Bins)

    KQT = np.real(buildKQT(np.abs(K0),tg,T2Bins))

    ###############################################
    # Linear Inversion 
    ############################################### 
    print("Calling inversion", flush=True)
    inv, ibreak, errn, phim, phid, mkappa, Wd, Wm, alphastar = logBarrier(KQT, np.ravel(V), T2Bins, "lcurve", MAXITER=150, sigma=np.ravel(VS), alpha=1e6, smooth="Smallest" ) 


    ################################
    # Summary plots, Data Space    #
    ################################

    # TODO, need to clean this up for the case of multiple channels! Each channel should be a new row. It will be ugly, but important 
    # TODO, loop over channels 
    
    ich = 0     
    for ch in cont['data'][dat]['channels']:

        figx = plt.figure( figsize=(pc2in(42.0),pc2in(22.)) )
        ax1 = figx.add_axes([.100, .15, .200, .70])    
        ax2 = figx.add_axes([.325, .15, .200, .70])   # shifted to make room for shared colourbar 
        axc1= figx.add_axes([.550, .15, .025, .70])   # shifted to make room for shared colourbar 
        ax3 = figx.add_axes([.670, .15, .200, .70])    
        axc2= figx.add_axes([.895, .15, .025, .70])   # shifted to make room for shared colourbar 

        ax3.set_yscale('log')
        ax2.set_yscale('log')
        ax1.set_yscale('log')
    
        ax2.yaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])
    
        ax3.set_xscale('log')
        ax2.set_xscale('log')
        ax1.set_xscale('log')

        ax1.set_ylabel("Q (A $\cdot$ s)")
        ax1.set_xlabel("time (s)")
        ax2.set_xlabel("time (s)")
        ax3.set_xlabel("time (s)")

        #TT, QQQ = np.meshgrid(tg, np.ravel(QQ))
        
        TT, QQQ = np.meshgrid(tg, np.ravel(QQ[ich]))
        nq = np.shape(QQ[ich])[0] - 1 # to account for padding in pcolor 
        nt = np.shape(tg)[0]
        ntq = nt*nq
        
        VV = V[ich*nq:ich*nq+nq,:]   # slice this channel
        VVS = VS[ich*nq:ich*nq+nq,:] # slice this channel

        mmax = np.max(np.abs(VV))
        mmin = np.min(VV)

        obs = ax1.pcolor(TT, QQQ, VV, cmap=cmocean.cm.curl_r, vmin=-mmax, vmax=mmax, shading='auto')  # pcolor edge not defined 
        ax1.set_title("observed")
 
        pre = np.dot(KQT[ich*ntq:(ich+1)*ntq,:], inv)
 
        PRE = np.reshape( pre, np.shape(VV)  )
        prem = ax2.pcolor(TT, QQQ, PRE, cmap=cmocean.cm.curl_r, vmin=-mmax, vmax=mmax,shading='auto' )
        ax2.set_title("predicted")

        cbar = plt.colorbar(prem, axc1)
        axc1.set_ylim( [np.min(VV), np.max(VV)] )
        cbar.outline.set_edgecolor(None)
        cbar.set_label('$V_N$ (nV)')

        DIFF = (PRE-VV) / VVS
        md = np.max(np.abs(DIFF))
        dim = ax3.pcolor(TT, QQQ, DIFF, cmap=cmocean.cm.balance, vmin=-md, vmax=md, shading='auto')
        ax3.set_title("misfit / $\widehat{\sigma}$")
    
        cbar2 = plt.colorbar(dim, axc2)
        #axc1.set_ylim( [np.min(V), np.max(V)] )
        cbar2.outline.set_edgecolor(None)
        cbar2.set_label('$V_N$ (nV)')
        #plt.colorbar(dim, ax3)
    
        figx.suptitle(ch + " linear Inversion")
        plt.savefig(ch + "dataspace.pdf")

        ich += 1

    ###############################################
    # Non-linear refinement! 
    ###############################################   

    nonLinearRefinement = cont['NonLinearRefinement']
    if nonLinearRefinement: 

        KQTc = buildKQT(K0, tg, T2Bins)
        prec = np.abs(np.dot(KQTc, inv))
        phidc = np.linalg.norm(np.dot(Wd,prec-np.ravel(V)))**2
        print("PHID forward linear=", errn, "PHID forward nonlinear=", phidc/len(np.ravel(V)))
    
        res = nl.nonlinearinversion(inv, Wd, KQTc, np.ravel(V), Wm, alphastar )   
        if res.success == True:    
            INVc = np.reshape(res.x, (len(ifaces)-1,cont["T2Bins"]["number"]) )
            prec = np.abs(np.dot(KQTc, res.x))
            phidc = np.linalg.norm(np.dot(Wd,prec-np.ravel(V)))**2
            PREc = np.reshape( prec, np.shape(V)  )
            print("PHID linear=", errn, "PHID nonlinear=", phidc/len(np.ravel(V)))

        while phidc/len(np.ravel(V)) > errn:  
            phidc_old = phidc/len(np.ravel(V))
            #alphastar *= .9 
            res = nl.nonlinearinversion(res.x, Wd, KQTc, np.ravel(V), Wm, alphastar )   
            if res.success == True:    
                INVc = np.reshape(res.x, (len(ifaces)-1,cont["T2Bins"]["number"]) )
                prec = np.abs(np.dot(KQTc, res.x))
                phidc = np.linalg.norm(np.dot(Wd,prec-np.ravel(V)))**2
                PREc = np.reshape( prec, np.shape(V)  )
                print("PHID linear=", errn, "PHID nonlinear=", phidc/len(np.ravel(V)))
            else:
                break

            if phidc_old - phidc/len(np.ravel(V)) < 0.005:
                print("Not making progress reducing misfit in nonlinear refinement")
                break

        # Turn this into a nice figure w/ shared axes etc.    
 
#         plt.matshow(PREc, cmap='Blues')
#         plt.gca().set_title("nonlinear predicted")
#         plt.colorbar()
# 
#         DIFFc = (PREc-V) / VS
#         md = np.max(np.abs(DIFF))
#         plt.matshow(DIFFc, cmap=cmocean.cm.balance, vmin=-md, vmax=md)
#         plt.gca().set_title("nonlinear misfit / $\widehat{\sigma}$")
#         plt.colorbar()


        ################################
        # Summary plots, Data Space    #
        ################################

    
        ich = 0     
        for ch in cont['data'][dat]['channels']:

            figx = plt.figure( figsize=(pc2in(42.0),pc2in(22.)) )
            ax1 = figx.add_axes([.100, .15, .200, .70])    
            ax2 = figx.add_axes([.325, .15, .200, .70])   # shifted to make room for shared colourbar 
            axc1= figx.add_axes([.550, .15, .025, .70])   # shifted to make room for shared colourbar 
            ax3 = figx.add_axes([.670, .15, .200, .70])    
            axc2= figx.add_axes([.895, .15, .025, .70])   # shifted to make room for shared colourbar 

            ax3.set_yscale('log')
            ax2.set_yscale('log')
            ax1.set_yscale('log')
    
            ax2.yaxis.set_ticklabels([])
            ax3.yaxis.set_ticklabels([])
    
            ax3.set_xscale('log')
            ax2.set_xscale('log')
            ax1.set_xscale('log')

            ax1.set_ylabel("Q (A $\cdot$ s)")
            ax1.set_xlabel("time (s)")
            ax2.set_xlabel("time (s)")
            ax3.set_xlabel("time (s)")

            #TT, QQQ = np.meshgrid(tg, np.ravel(QQ))
        
            TT, QQQ = np.meshgrid(tg, np.ravel(QQ[ich]))
            nq = np.shape(QQ[ich])[0] - 1 # to account for padding in pcolor 
            nt = np.shape(tg)[0]
            ntq = nt*nq
        
            VV = V[ich*nq:ich*nq+nq,:]   # slice this channel
            VVS = VS[ich*nq:ich*nq+nq,:] # slice this channel

            mmax = np.max(np.abs(VV))
            mmin = np.min(VV)

            obs = ax1.pcolor(TT, QQQ, VV, cmap=cmocean.cm.curl_r, vmin=-mmax, vmax=mmax, shading='auto')
            ax1.set_title("observed")

            ## Here neds to change  
            pre = np.abs(np.dot(KQTc[ich*ntq:(ich+1)*ntq,:], inv))
 
            PRE = np.reshape( pre, np.shape(VV)  )
            prem = ax2.pcolor(TT, QQQ, PRE, cmap=cmocean.cm.curl_r, vmin=-mmax, vmax=mmax, shading='auto' )
            ax2.set_title("predicted")

            cbar = plt.colorbar(prem, axc1)
            axc1.set_ylim( [np.min(VV), np.max(VV)] )
            cbar.outline.set_edgecolor(None)
            cbar.set_label('$V_N$ (nV)')

            DIFF = (PRE-VV) / VVS
            md = np.max(np.abs(DIFF))
            dim = ax3.pcolor(TT, QQQ, DIFF, cmap=cmocean.cm.balance, vmin=-md, vmax=md, shading='auto')
            ax3.set_title("misfit / $\widehat{\sigma}$")
    
            cbar2 = plt.colorbar(dim, axc2)
            #axc1.set_ylim( [np.min(V), np.max(V)] )
            cbar2.outline.set_edgecolor(None)
            cbar2.set_label('$V_N$ (nV)')
            #plt.colorbar(dim, ax3)
    
            figx.suptitle(ch + " non-linear Inversion")
        
            plt.savefig(ch + "_NLdataspace.pdf")

            ich += 1



 
    ###############################################
    # Appraise DOI using simplified MRM 
    ###############################################

    CalcDOI = cont['CalcDOI']

    if CalcDOI:
    
        pdf = PdfPages('resolution_analysis' + '.pdf' )
        MRM = np.zeros((len(ifaces)-1, len(ifaces)-1))

        # Build delta models 
        DELTA = []
        
        for ilay in range(len(ifaces)-1):
        #for ilay in range(4):
            iDeltaT2 = len(T2Bins)//2
            deltaMod = np.zeros( (len(ifaces)-1, len(T2Bins)) )
            deltaMod[ilay][iDeltaT2] = 0.3
            dV = np.dot(KQT, np.ravel(deltaMod))
            #dinv, dibreak, derrn = logBarrier( KQT, dV, T2Bins, "single", MAXITER=1, sigma=np.ravel(VS), alpha=alphastar, smooth="Smallest" ) 
            #output = invertDelta(KQT, dV, T2Bins, np.ravel(VS), alphastar)
            DELTA.append(dV)

        print("Performing resolution analysis in parallel, printed output may not be inorder.", flush=True) 
        with multiprocessing.Pool() as pool: 
            invresults = pool.starmap(invertDelta, zip(itertools.repeat(KQT), DELTA, itertools.repeat(T2Bins), itertools.repeat(np.ravel(VS)), itertools.repeat(alphastar) )) 
        #    invresults = pool.starmap(logBarrier, zip(itertools.repeat(KQT), DELTA, itertools.repeat(T2Bins), itertools.repeat('single'), \
        #        itertools.repeat('MAXITER=1'), itertools.repeat(np.ravel(VS)), itertools.repeat(alphastar))) #, itertools.repeat(u'smooth=\'Smallest\'')) ) 
 
        # This could be parallelized 
        for ilay in range(len(ifaces)-1):

            # invert 
            #dinv, dibreak, derrn = logBarrier(KQT, dV, T2Bins, "single", MAXITER=1, sigma=np.ravel(VS), alpha=alphastar, smooth="Smallest" ) 
            #print("Sum dinv from", str(ifaces[ilay]), "to", str(ifaces[ilay+1]), "=", np.sum(dinv))
            dinv, dibreak, derrn = invresults[ilay] 

    
            DINV = np.reshape(dinv, (len(ifaces)-1,cont["T2Bins"]["number"]) )
            MRM[ilay,:] = np.sum(DINV, axis=1)

            Y,X = meshgrid( ifaces, T2Bins2 )
            fig = plt.figure( figsize=(pc2in(20.0),pc2in(22.)) )
            ax1 = fig.add_axes( [.2,.15,.6,.7] )
            im = ax1.pcolor(X, Y, DINV.T, cmap=cmocean.cm.tempo, shading='auto')
            ax1.plot( T2Bins[iDeltaT2], (ifaces[ilay]+ifaces[ilay+1])/2, 's', markersize=6, markeredgecolor='black') #, markerfacecolor=None )  
            im.set_edgecolor('face')
            ax1.set_xlabel(u"$T_2^*$ (ms)")
            ax1.set_ylabel(u"depth (m)")
            ax1.set_xlim( T2Bins2[0], T2Bins2[-1] )
            ax1.set_ylim( ifaces[-1], ifaces[0] )

            ax2 = ax1.twiny()
            ax2.plot( np.sum(DINV, axis=1), (ifaces[1:]+ifaces[0:-1])/2 ,  color='red' )
            ax2.set_xlabel(u"total water (m$^3$/m$^3$)")
            ax2.set_ylim( ifaces[-1], ifaces[0] )
            ax2.xaxis.set_major_locator( MaxNLocator(nbins = 3) )   
            ax2.get_xaxis().set_major_formatter(FormatStrFormatter('%0.2f'))

            pdf.savefig(facecolor=[0,0,0,0])
            plt.close(fig)

        np.save("MRM", MRM)
        centres = (ifaces[0:-1]+ifaces[1:])/2
        X,Y = np.meshgrid(ifaces,ifaces)

        fig = plt.figure( figsize=(pc2in(20.0),pc2in(22.)) )
        ax1 = fig.add_axes( [.2,.15,.6,.7] )
        ax1.pcolor(X,Y,MRM, cmap = cmocean.cm.ice)
        ax1.set_ylim(ifaces[-1], ifaces[0])
        maxDepth = np.argmax(MRM, axis=0)

        plt.plot(centres[maxDepth], centres, color='white')

        # Determine DOI 
        DOIMetric =  centres[maxDepth]/centres #> 0.9 
        DOI = ifaces[ np.where(DOIMetric < 0.9 ) ][0]
        plt.axhline(y=DOI, color='white', linestyle='-.')

        ax1.set_ylim( ifaces[-1], ifaces[0] )
        ax1.set_xlim( ifaces[0], ifaces[-1] )
        ax1.set_xlabel(u"depth (m)")
        ax1.set_ylabel(u"depth (m)")
        plt.savefig("resolutionmatrix.pdf")
        pdf.close()

    INV = np.reshape(inv, (len(ifaces)-1,cont["T2Bins"]["number"]) )


    ##############  LINEAR RESULT   ##########################

    Y,X = meshgrid( ifaces, T2Bins2 )
    fig = plt.figure( figsize=(pc2in(20.0),pc2in(22.)) )
    ax1 = fig.add_axes( [.2,.15,.6,.7] )
    im = ax1.pcolor(X, Y, INV.T, cmap=cmocean.cm.tempo) #cmap='viridis')
    im.set_edgecolor('face')
    ax1.set_xlim( T2Bins[0], T2Bins[-1] )
    ax1.set_ylim( ifaces[-1], ifaces[0] )
    cb = plt.colorbar(im, label = u"PWC (m$^3$/m$^3$)") #, format='%1.1f')
    cb.locator = MaxNLocator( nbins = 4)
    cb.ax.yaxis.set_offset_position('left')                         
    cb.update_ticks()
 
    ax1.set_xlabel(u"$T_2^*$ (ms)")
    ax1.set_ylabel(u"depth (m)")
    
    ax1.get_xaxis().set_major_formatter(FormatStrFormatter('%1.0f'))
    ax1.get_yaxis().set_major_formatter(FormatStrFormatter('%1.0f'))
    ax1.xaxis.set_major_locator( MaxNLocator(nbins = 4) )   


    ax2 = ax1.twiny()
    ax2.plot( np.sum(INV, axis=1), (ifaces[1:]+ifaces[0:-1])/2 ,  color='red' )
    ax2.set_xlabel(u"total water (m$^3$/m$^3$)")
    ax2.set_ylim( ifaces[-1], ifaces[0] )
    ax2.xaxis.set_major_locator( MaxNLocator(nbins = 3) )   
    ax2.get_xaxis().set_major_formatter(FormatStrFormatter('%0.2f'))
    #ax2.axhline( y=ifaces[SNRidx], xmin=0, xmax=1, color='black', linestyle='dashed'  )
    if CalcDOI:
        ax2.axhline( y=DOI, xmin=0, xmax=1, color='black', linestyle='dashed'  )

    plt.savefig("akvoInversion.pdf")

    #############
    # water plot#

    fig2 = plt.figure( figsize=(pc2in(20.0),pc2in(22.)) )
    ax = fig2.add_axes( [.2,.15,.6,.7] )
    
    # Bound water cutoff 
    Bidx = T2Bins<33.0
    twater = np.sum(INV, axis=1)
    bwater = np.sum(INV[:,Bidx], axis=1)
    
    ax.plot( twater, (ifaces[0:-1]+ifaces[1::])/2, label="NMR total water", color='blue' )
    ax.plot( bwater, (ifaces[0:-1]+ifaces[1::])/2, label="NMR bound water", color='green' )
    
    ax.fill_betweenx((ifaces[0:-1]+ifaces[1::])/2 , twater, bwater, where=twater >= bwater, facecolor='blue', alpha=.5)
    ax.fill_betweenx((ifaces[0:-1]+ifaces[1::])/2 , bwater, 0, where=bwater >= 0, facecolor='green', alpha=.5)
    
    ax.set_xlabel(r"$\theta_N$ (m$^3$/m$^3$)")
    ax.set_ylabel(r"depth (m)")
    
    ax.set_ylim( ifaces[-1], ifaces[0] )
    ax.set_xlim( 0, ax.get_xlim()[1] )
    
    #ax.axhline( y=ifaces[SNRidx], xmin=0, xmax=1, color='black', linestyle='dashed'  )
    if CalcDOI:
        ax.axhline( y=DOI, xmin=0, xmax=1, color='black', linestyle='dashed'  )

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')   
 
    plt.savefig("akvoInversionWC.pdf")
    plt.legend()  
    
    fr = pd.DataFrame( INV, columns=T2Bins ) #[0:-1] )
    fr.insert(0, "layer top", ifaces[0:-1] )
    fr.insert(1, "layer bottom", ifaces[1::] )
    fr.insert(2, "NMR total water", np.sum(INV, axis=1) )
    fr.insert(3, "NMR bound water", bwater )
    fr.insert(4, "Layer SNR", SNR )
    if CalcDOI:
        fr.insert(5, "Resolution", DOIMetric )

    fr.to_csv("akvoInversion.csv", mode='w+')    


    ##############  NONLINEAR RESULT   ##########################

    if nonLinearRefinement: 
        Y,X = meshgrid( ifaces, T2Bins2 )
        fig = plt.figure( figsize=(pc2in(20.0),pc2in(22.)) )
        ax1 = fig.add_axes( [.2,.15,.6,.7] )
        im = ax1.pcolor(X, Y, INVc.T, cmap=cmocean.cm.tempo) #cmap='viridis')
        #im = ax1.pcolor(X[0:SNRidx,:], Y[0:SNRidx,:], INV.T[0:SNRidx,:], cmap=cmocean.cm.tempo) #cmap='viridis')
        #im = ax1.pcolor(X[SNRidx::,:], Y[SNRidx::,:], INV.T[SNRidx::,:], cmap=cmocean.cm.tempo, alpha=.5) #cmap='viridis')
        #im = ax1.pcolormesh(X, Y, INV.T, alpha=alphas) #, cmap=cmocean.cm.tempo) #cmap='viridis')
        #im = ax1.pcolormesh(X, Y, INV.T, alpha=alphas) #, cmap=cmocean.cm.tempo) #cmap='viridis')
        #ax1.axhline( y=ifaces[SNRidx], xmin=T2Bins[0], xmax=T2Bins[-1], color='black'  )
        im.set_edgecolor('face')
        ax1.set_xlim( T2Bins[0], T2Bins[-1] )
        ax1.set_ylim( ifaces[-1], ifaces[0] )
        cb = plt.colorbar(im, label = u"PWC (m$^3$/m$^3$)") #, format='%1.1f')
        cb.locator = MaxNLocator( nbins = 4)
        cb.ax.yaxis.set_offset_position('left')                         
        cb.update_ticks()
 
        ax1.set_xlabel(u"$T_2^*$ (ms)")
        ax1.set_ylabel(u"depth (m)")
    
        ax1.get_xaxis().set_major_formatter(FormatStrFormatter('%1.0f'))
        ax1.get_yaxis().set_major_formatter(FormatStrFormatter('%1.0f'))
        ax1.xaxis.set_major_locator( MaxNLocator(nbins = 4) )   

        #ax1.xaxis.set_label_position('top') 

        ax2 = ax1.twiny()
        ax2.plot( np.sum(INVc, axis=1), (ifaces[1:]+ifaces[0:-1])/2 ,  color='red' )
        ax2.set_xlabel(u"total water (m$^3$/m$^3$)")
        ax2.set_ylim( ifaces[-1], ifaces[0] )
        ax2.xaxis.set_major_locator( MaxNLocator(nbins = 3) )   
        ax2.get_xaxis().set_major_formatter(FormatStrFormatter('%0.2f'))
        #ax2.axhline( y=ifaces[SNRidx], xmin=0, xmax=1, color='black', linestyle='dashed'  )
        if CalcDOI:
            ax2.axhline( y=DOI, xmin=0, xmax=1, color='black', linestyle='dashed'  )
        #ax2.xaxis.set_label_position('bottom') 
        #fig.suptitle("Non linear inversion")
        plt.savefig("akvoInversionNL.pdf")



        #############
        # water plot#

        fig2 = plt.figure( figsize=(pc2in(20.0),pc2in(22.)) )
        ax = fig2.add_axes( [.2,.15,.6,.7] )
    
        # Bound water cutoff 
        Bidx = T2Bins<33.0
        twater = np.sum(INVc, axis=1)
        bwater = np.sum(INVc[:,Bidx], axis=1)
    
        ax.plot( twater, (ifaces[0:-1]+ifaces[1::])/2, label="NMR total water", color='blue' )
        ax.plot( bwater, (ifaces[0:-1]+ifaces[1::])/2, label="NMR bound water", color='green' )
    
        ax.fill_betweenx((ifaces[0:-1]+ifaces[1::])/2 , twater, bwater, where=twater >= bwater, facecolor='blue', alpha=.5)
        ax.fill_betweenx((ifaces[0:-1]+ifaces[1::])/2 , bwater, 0, where=bwater >= 0, facecolor='green', alpha=.5)
    
        ax.set_xlabel(r"$\theta_N$ (m$^3$/m$^3$)")
        ax.set_ylabel(r"depth (m)")
    
        ax.set_ylim( ifaces[-1], ifaces[0] )
        ax.set_xlim( 0, ax.get_xlim()[1] )
    
        #ax.axhline( y=ifaces[SNRidx], xmin=0, xmax=1, color='black', linestyle='dashed'  )
        if CalcDOI:
            ax.axhline( y=DOI, xmin=0, xmax=1, color='black', linestyle='dashed'  )
    
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')   
    
        plt.savefig("akvoNLInversionWC.pdf")
        plt.legend()  


        # Report results into a text file 
        fr = pd.DataFrame( INVc, columns=T2Bins ) #[0:-1] )
        fr.insert(0, "layer top", ifaces[0:-1] )
        fr.insert(1, "layer bottom", ifaces[1::] )
        fr.insert(2, "NMR total water", np.sum(INVc, axis=1) )
        fr.insert(3, "NMR bound water", bwater )
        fr.insert(4, "Layer SNR", SNR )
        if CalcDOI:
            fr.insert(5, "Resolution", DOIMetric )

        fr.to_csv("akvoNLInversion.csv", mode='w+')    
        #fr.to_excel("akvoInversion.xlsx")    

 
    plt.show()

if __name__ == "__main__":
    main() 

