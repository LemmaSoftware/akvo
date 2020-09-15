from PyQt5.QtCore import *
import numpy as np
import scipy.signal as signal
import pylab
import sys
import scipy
from scipy import stats
import copy
import struct
from scipy.io.matlab import mio
from numpy import pi
from math import floor
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker 
from matplotlib.ticker import MaxNLocator

import seaborn as sns 
from akvo.tressel.SlidesPlot import deSpine

import multiprocessing 
import itertools 

import padasip as pa

import akvo.tressel.adapt as adapt
#import akvo.tressel.cadapt as adapt # cython for more faster
import akvo.tressel.decay as decay
import akvo.tressel.pca as pca
import akvo.tressel.rotate as rotate
import akvo.tressel.cmaps as cmaps
import akvo.tressel.harmonic as harmonic

import cmocean # colormaps for geophysical data 
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='inferno', cmap=cmaps.inferno)
plt.register_cmap(name='inferno_r', cmap=cmaps.inferno_r)

plt.register_cmap(name='magma', cmap=cmaps.magma)
plt.register_cmap(name='magma_r', cmap=cmaps.magma_r)


def xxloadGMRBinaryFID( rawfname, info ):
    """ Reads a single binary GMR file and fills into DATADICT
    """

    #################################################################################
    # figure out key data indices
    # Pulse        
    nps  = (int)((info["prePulseDelay"])*info["samp"])
    npul   = (int)(self.pulseLength[0]*self.samp) #+ 100 

    # Data 
    nds  = nps+npul+(int)((self.deadTime)*self.samp);        # indice pulse 1 data starts 
    nd1 = (int)(1.*self.samp)                                # samples in first pulse

    invGain = 1./self.RxGain        
    invCGain = self.CurrentGain        

    pulse = "Pulse 1"
    chan = self.DATADICT[pulse]["chan"] 
    rchan = self.DATADICT[pulse]["rchan"] 
        
    rawFile = open( rawfname, 'rb')
        
    T = N_samp * self.dt 
    TIMES = np.arange(0, T, self.dt) - .0002 # small offset in GMR DAQ?

    for ipm in range(self.nPulseMoments):
        buf1 = rawFile.read(4)
        buf2 = rawFile.read(4)
                
        N_chan = struct.unpack('>i', buf1 )[0]
        N_samp = struct.unpack('>i', buf2 )[0]

        DATA = np.zeros([N_samp, N_chan+1])
        for ichan in range(N_chan):
            DATADUMP = rawFile.read(4*N_samp)
            for irec in range(N_samp):
                DATA[irec,ichan] = struct.unpack('>f', DATADUMP[irec*4:irec*4+4])[0]
        
    return DATA, TIMES

class SNMRDataProcessor(QObject):
    """ Revised class for preprocessing sNMR Data. 
        Derived types can read GMR files  
    """ 
    def __init__(self):
        QObject.__init__(self)
        self.numberOfMoments            = 0
        self.numberOfPulsesPerMoment    = 0
        self.pulseType                  = "NONE"
        self.transFreq                  = 0
        self.pulseLength                = np.zeros(1)
        self.nPulseMoments              = 0
        self.dt                         = 0

    def mfreqz(self, b,a=1):
        """ Plots the frequency response of a filter specified with a and b weights
        """
        import scipy.signal as signal
        pylab.figure(124)
        w,h = signal.freqz(b,a)
        w /= max(w)
        w *= .5/self.dt
        h_dB = 20 * pylab.log10 (abs(h))
        pylab.subplot(211)
        #pylab.plot(w/max(w),h_dB)
        pylab.plot(w,h_dB)
        pylab.ylim(-150, 5)
        pylab.ylabel('Magnitude (dB)')
        #pylab.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
        pylab.xlabel(r'Hz')
        pylab.title(r'Frequency response')
        pylab.subplot(212)
        h_Phase = pylab.unwrap(pylab.arctan2(pylab.imag(h), pylab.real(h)))
        #pylab.plot(w/max(w),h_Phase)
        pylab.plot(w,h_Phase)
        pylab.ylabel('Phase (radians)')
        pylab.xlabel(r'Hz')
        #pylab.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
        pylab.title(r'Phase response')
        pylab.subplots_adjust(hspace=0.5)

    def mfreqz2(self, b, a, canvas):
        "for analysing filt-filt"
        import scipy.signal as signal
        canvas.reAx2(False,False)

        canvas.ax1.tick_params(axis='both', which='major', labelsize=8)
        canvas.ax2.tick_params(axis='both', which='major', labelsize=8)
        #canvas.ax2.tick_params(axis='both', which='minor', labelsize=6)

        #pylab.figure(124)
        w,h = signal.freqz(b,a)
        w /= max(w)
        w *= .5/self.dt
        h_dB = 20 * pylab.log10(abs(h*h) + 1e-16)
        #ab.subplot(211)
        #pylab.plot(w/max(w),h_dB)
        canvas.ax1.plot(w,h_dB)
        canvas.ax1.set_ylim(-150, 5)
        canvas.ax1.set_ylabel('Magnitude [db]', fontsize=8)
        #pylab.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
        canvas.ax1.set_xlabel(r'[Hz]', fontsize=8)
        canvas.ax1.set_title(r'Frequency response', fontsize=8)
        canvas.ax1.grid(True)

        tt = np.arange(0, .02, self.dt)
        impulse = signal.dimpulse((self.filt_z, self.filt_p, self.filt_k, self.dt), t=tt)
        #impulse = signal.dstep((self.filt_z, self.filt_p, self.filt_k, self.dt), t=tt)
        #print impulse 
        #for ii in range(len(impulse[1])):
        impulse_dB  = 20.*np.log10(np.abs(np.array(impulse[1][0])))
        #canvas.ax2.plot(np.array(impulse[0]), impulse_dB)
        canvas.ax2.plot(np.array(impulse[0]), impulse[1][0])
        #h_Phase = pylab.unwrap(pylab.arctan2(pylab.imag(h), pylab.real(h)))
        #canvas.ax2.plot(w,h_Phase)
        canvas.ax2.set_ylabel('response [%]', fontsize=8)
        canvas.ax2.set_xlabel(r'time [s]', fontsize=8)
        canvas.ax2.set_title(r'impulse response', fontsize=8)
        #canvas.ax2.grid(True) 
        canvas.draw()
        # search for last  
        return  impulse #[np.where(impulse[1][0]  > .01)[-1]]
        

class GMRDataProcessor(SNMRDataProcessor):

    # slots 
    progressTrigger = pyqtSignal("int")
    doneTrigger = pyqtSignal()
    enableDSPTrigger = pyqtSignal()
    updateProcTrigger = pyqtSignal()

    def __init__(self):

        SNMRDataProcessor.__init__(self)
        self.maxBusV                = 0.
        self.samp                   = 50000.         # sampling frequency
        self.dt                     = 2e-5           # sampling rate 
        self.deadTime               = .0055          # instrument dead time before measurement
        self.prePulseDelay          = 0.05           # delay before pulse
        self.windead                = 0.             # FD window filter dead time
        self.pulseType              = -1
        self.transFreq              = -1
        self.maxBusV                = -1
        self.pulseLength            = -1
        self.interpulseDelay        = -1             # for T2, Spin Echo
        self.repetitionDelay        = -1             # delay between first pulse
        self.nPulseMoments          = -1             # Number of pulse moments per stack
        self.TuneCapacitance        = -1             # tuning capac in uF
        self.nTransVersion          = -1             # Transmitter version
        self.nDAQVersion            = -1             # DAQ software version 
        self.nInterleaves           = -1             # num interleaves
#        self.nReceiveChannels       = 4              # Num receive channels

        self.RotatedAmplitude = False
#        self.DATA                   = np.zeros(1) # Numpy array to hold all data, dimensions resized based on experiment
#        self.PULSES                 = np.zeros(1) # Numpy array to hold all data, dimensions resized based on experiment
 
    def Print(self):
        print ("pulse type", self.pulseType)
        print ("maxBusV", self.maxBusV)
        print ("inner pulse delay", self.interpulseDelay) 
        print ("tuning capacitance", self.TuneCapacitance)      
        print ("sampling rate", self.samp)         
        print ("dt", self.dt)            
        print ("dead time", self.deadTime)     
        print ("pre pulse delay", self.prePulseDelay)     
        print ("number of pulse moments", self.nPulseMoments)
        print ("pulse Length", self.pulseLength) 
        print ("trans freq", self.transFreq)

    def readHeaderFile(self, FileName):

        HEADER = np.loadtxt(FileName)

        pulseTypeDict = {
            1 : lambda: "FID",
            2 : lambda: "T1",
            3 : lambda: "SPINECHO",
            4 : lambda: "4PhaseT1"
        }
        
        pulseLengthDict = {
            1 : lambda x: np.ones(1) * x,
            2 : lambda x: np.ones(2) * x,
            3 : lambda x: np.array([x, 2.*x]),
            4 : lambda x: np.ones(2) * x
        }

        self.pulseType       = pulseTypeDict.get((int)(HEADER[0]))()
        self.transFreq       = HEADER[1]
        self.maxBusV         = HEADER[2]
        self.pulseLength     = pulseLengthDict.get((int)(HEADER[0]))(1e-3*HEADER[3])
        self.interpulseDelay = 1e-3*HEADER[4]      # for T2, Spin Echo
        self.repetitionDelay = HEADER[5]           # delay between first pulse
        self.nPulseMoments   = (int)(HEADER[6])    # Number of pulse moments per stack
        self.TuneCapacitance = HEADER[7]           # tuning capacitance in uF
        self.nTransVersion   = HEADER[8]           # Transmitter version
        self.nDAQVersion     = HEADER[9]           # DAQ software version 
        self.nInterleaves    = HEADER[10]          # num interleaves

        self.gain()
        
        # default 
        self.samp                   = 50000.        # sampling frequency
        self.dt                     = 2e-5          # sampling rate 

        # newer header files contain 64 entries
        if self.nDAQVersion >= 2:
           #self.deadtime       = HEADER[11]
           #self.unknown        = HEADER[12]
           #self.PreAmpGain     = HEADER[13]
            self.samp           = HEADER[14]     # sampling frequency
            self.dt             = 1./self.samp   # sampling rate 
            self.deadTime       = .0055          # instrument dead time before measurement
            self.prePulseDelay  = 0.05           # delay before pulse
            #exit() 

    def gain(self):

        #######################################################
        # Circuit gain
        # From MRSMatlab 
        w = 2*np.pi*self.transFreq
        # 1e6 due to uF of reported capacitance
        L_coil = 1e6/(self.TuneCapacitance*(w**2))
        R_coil = 1.
        Z1_in = .5 + 1j*.5*w
        Z2_in = 1./(1j*w*.000001616) 
        Z_eq_inv = (1./Z1_in) + (1./Z2_in)
        Zeq = 1./Z_eq_inv
        Zsource = R_coil + 1j*w*L_coil
        voltage_in = Zeq / (Zsource + Zeq)
        self.circuitGain = np.abs(voltage_in)
        self.circuitPhase_deg = (180/np.pi)+np.angle(voltage_in)
        circuitImpedance_ohms = np.abs(Zsource + Zeq)

        ######################################################
        # PreAmp gain
        if self.nTransVersion == 4:
            self.PreAmpGain = 1000.
        elif self.nTransVersion == 1 or self.nTransVersion == 2 or self.nTransVersion == 3 or self.nTransVersion == 6:
            self.PreAmpGain = 500.
        else:
            print ("unsupported transmitter version")
            exit(1)

        # Total Receiver Gain
        self.RxGain = self.circuitGain * self.PreAmpGain

        #####################################################
        # Current gain
        if floor(self.nDAQVersion) == 1:
            self.CurrentGain = 150.
        elif floor(self.nDAQVersion) == 2:
            self.CurrentGain = 180.

    def updateProgress(self):
        pass   
 
    def TDSmartStack(self, outlierTest, MADcutoff, canvas):
        fs = 10 # fontsize 
        #print("Line 300 in mrsurvey")
        Stack = {}
        # align for stacking and modulate
        for pulse in self.DATADICT["PULSES"]:
            stack = np.zeros(( len(self.DATADICT[pulse]["chan"]), self.DATADICT["nPulseMoments"],\
                               len(self.DATADICT["stacks"]), len(self.DATADICT[pulse]["TIMES"]) ))
            for ipm in range(self.DATADICT["nPulseMoments"]):
                istack = 0
                for sstack in self.DATADICT["stacks"]:
                    if self.pulseType == "FID" or pulse == "Pulse 2":
                        if floor(self.nDAQVersion) < 2:
                            mod = 1
                        else:
                            mod = (-1.)**(ipm%2) * (-1.)**(sstack%2)
                    elif self.pulseType == "T1":
                        #mod = (-1.)**(sstack%2)
                        #mod = (-1)**(ipm%2) * (-1)**(sstack%2)
                        #mod = (-1)**(ipm%2) * (-1.**(((sstack-1)/2)%2))
                        #print("mod", mod, ipm, sstack,  (-1.)**(ipm%2),  -1.0**(((sstack-1)/2)%2 ))
                        #mod = (-1.)**((ipm+1)%2) * (-1.**(((sstack)/2)%2))
                        #mod = (-1.)**((ipm-1)%2) * (-1.)**((sstack-1)%2)
                        #mod = 1 # (-1.**(((sstack-1)/2)%2))

                        # These two give great noise estimate
                        #qcycler = np.array([1,-1,-1,1])
                        #scycler = np.array([1,-1,1,-1])

                        qcycler = np.array([ 1, 1])
                        scycler = np.array([ 1, 1])
                        mod = qcycler.take([ipm], mode='wrap')*scycler.take([sstack], mode='wrap')
                        #mod = (-1.)**(ipm%2) * (-1.)**(sstack%2)
                    elif self.pulseType == "4PhaseT1":
                        mod = (-1.)**(ipm%2) * (-1.**(((sstack-1)/2)%2))
                    ichan = 0
                    for chan in self.DATADICT[pulse]["chan"]:
                        stack[ichan,ipm,istack,:] += mod*self.DATADICT[pulse][chan][ipm][sstack]
                        ichan += 1
                    istack += 1
            Stack[pulse] = stack
        
        ######################################### 
        # simple stack and plot of simple stack #
        #########################################
        canvas.reAxH2(np.shape(stack)[0], False, False)
        axes = canvas.fig.axes
        SimpleStack = {}
        VarStack = {}
        for pulse in self.DATADICT["PULSES"]:
            SimpleStack[pulse] = {}
            VarStack[pulse] = {}
            ichan = 0
            for chan in self.DATADICT[pulse]["chan"]: 
                SimpleStack[pulse][chan] = 1e9*np.average( Stack[pulse][ichan], 1 ) 
                VarStack[pulse][chan] = 1e9*np.std( Stack[pulse][ichan], 1 ) 
                ax1 = axes[ 2*ichan ]
                #ax1.get_yaxis().get_major_formatter().set_useOffset(False)

                y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
                ax1.yaxis.set_major_formatter(y_formatter)

                ax1.plot( 1e3*self.DATADICT[pulse]["TIMES"], np.average(  SimpleStack[pulse][chan], 0 )) #, color='darkblue' )
                ax1.set_title("Ch." + str(chan) + ": avg FID", fontsize=fs)
                ax1.set_xlabel(r"time (ms)", fontsize=fs)

                if ichan == 0:
                    ax1.set_ylabel(r"signal (nV)", fontsize=fs)
                else:
                    plt.setp(ax1.get_yticklabels(), visible=False)
                    plt.setp(ax1.get_yaxis().get_offset_text(), visible=False) 
#                 if ichan == 1:
#                     canvas.ax2.plot( 1e3*self.DATADICT[pulse]["TIMES"], np.average(  SimpleStack[pulse][chan], 0 ), color='darkblue' )
#                     canvas.ax2.set_title("Ch." + str(chan) + ": total average FID", fontsize=8)
#                     canvas.ax2.set_xlabel(r"time [ms]", fontsize=8)
#                 if ichan == 2:
#                     canvas.ax3.plot( 1e3*self.DATADICT[pulse]["TIMES"], np.average(  SimpleStack[pulse][chan], 0 ), color='darkblue' )
#                     canvas.ax3.set_title("Ch." + str(chan) + ": total average FID", fontsize=8)
#                     canvas.ax3.set_xlabel(r"time [ms]", fontsize=8)
#                 if ichan == 3:
#                     canvas.ax4.plot( 1e3*self.DATADICT[pulse]["TIMES"], np.average(  SimpleStack[pulse][chan], 0 ), color='darkblue' )
#                     canvas.ax4.set_title("Ch." + str(chan) + ": total average FID", fontsize=8)
#                     canvas.ax4.set_xlabel(r"time [ms]", fontsize=8)
                ichan += 1

        #########################
        # Oulier rejectig stack #
        #########################
        if outlierTest == "MAD":
            MADStack = {}
            VarStack = {}
            #1.4826 is assumption of gaussian noise 
            madstack = np.zeros(( len(self.DATADICT[pulse]["chan"]),\
                                  self.DATADICT["nPulseMoments"], len(self.DATADICT[pulse]["TIMES"]) ))
            varstack = np.zeros(( len(self.DATADICT[pulse]["chan"]),\
                                  self.DATADICT["nPulseMoments"], len(self.DATADICT[pulse]["TIMES"]) ))
            for pulse in self.DATADICT["PULSES"]:
                MADStack[pulse] = {}
                VarStack[pulse] = {}
                ichan = 0
                for chan in self.DATADICT[pulse]["chan"]:
                    ax1 = axes[ 2*ichan  ]
                    for ipm in range(self.DATADICT["nPulseMoments"]):
#                         # brutal loop over time, can this be vectorized? 
#                         for it in range(len(self.DATADICT[pulse]["TIMES"])): 
#                             x = 1e9 *Stack[pulse][ichan,ipm,:,it]
#                             MAD = 1.4826 * np.median( np.abs(x-np.median(x)) )
#                             good = 0
#                             for istack in self.DATADICT["stacks"]:
#                                 if (np.abs(x[istack-1]-np.median(x))) / MAD < 2:
#                                     good += 1
#                                     madstack[ ichan, ipm, it ] += x[istack-1]
#                                 else:
#                                     pass
#                             madstack[ichan, ipm, it] /= good
#                         percent = int(1e2* (float)(ipm) / (float)(self.DATADICT["nPulseMoments"]) )
#                         self.progressTrigger.emit(percent)

                        # Vectorized version of above...much, much faster 
                        x = 1e9*copy.deepcopy(Stack[pulse][ichan][ipm,:,:])      # stack and time indices
                        tile_med =  np.tile( np.median(x, axis=0), (np.shape(x)[0],1)) 
                        MAD = MADcutoff * np.median(np.abs(x - tile_med), axis=0)
                        tile_MAD =  np.tile( MAD, (np.shape(x)[0],1)) 
                        good = np.abs(x-tile_med)/tile_MAD < 2. # 1.4826 # 2

                        madstack[ichan][ipm] = copy.deepcopy( np.ma.masked_array(x, good != True).mean(axis=0) )
                        varstack[ichan][ipm] = copy.deepcopy( np.ma.masked_array(x, good != True).std(axis=0) )
                        
                        # reporting
                        percent = int(1e2* (float)((ipm)+ichan*self.DATADICT["nPulseMoments"]) / 
                                           (float)(self.DATADICT["nPulseMoments"] * len(self.DATADICT[pulse]["chan"])))
                        self.progressTrigger.emit(percent)

                    ax1.plot( 1e3*self.DATADICT[pulse]["TIMES"], np.average(  madstack[ichan], 0  ))# , color='darkred')
                        
                    MADStack[pulse][chan] = madstack[ichan]
                    VarStack[pulse][chan] = varstack[ichan]
                    ichan += 1
 
            self.DATADICT["stack"] = MADStack 

        else:
            self.DATADICT["stack"] = SimpleStack 

        #########################################
        # Plot Fourier Transform representation #
        #########################################

#         canvas.fig.subplots_adjust(right=0.8)
#         cbar_ax = canvas.fig.add_axes([0.85, 0.1, 0.015, 0.355])
#         cbar_ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')                 
        im2 = []
        im1 = []
        for pulse in self.DATADICT["PULSES"]:
            ichan = 0
            axes = canvas.fig.axes
            vvmin = 1e10 
            vvmax = 0
            for chan in self.DATADICT[pulse]["chan"]: 
                ax1 = axes[2*ichan  ]
                ax2 = axes[2*ichan+1] # TODO fix hard coded number
                if outlierTest == "MAD":
                    X = np.fft.rfft( MADStack[pulse][chan][0,:] )
                    nu = np.fft.fftfreq(len( MADStack[pulse][chan][0,:]), d=self.dt)
                else:
                    X = np.fft.rfft( SimpleStack[pulse][chan][0,:] )
                    nu = np.fft.fftfreq(len( SimpleStack[pulse][chan][0,:]), d=self.dt)
                
                nu = nu[0:len(X)]
                nu[-1] = np.abs(nu[-1])
                df = nu[1] - nu[0]
                of = 0

                istart = int((self.transFreq-50.)/df)
                iend = int((self.transFreq+50.)/df)
                of = nu[istart]
                
                def freqlabel(xxx, pos):
                    return  '%1.0f' %(of + xxx*df)
                formatter = FuncFormatter(freqlabel)
        
                SFFT = np.zeros( (self.DATADICT["nPulseMoments"], len(X)), dtype=np.complex64 )
                SFFT[0,:] = X
                for ipm in range(1, self.DATADICT["nPulseMoments"]):
                    if outlierTest == "MAD":
                        SFFT[ipm,:] = np.fft.rfft( MADStack[pulse][chan][ipm,:] )
                    else:
                        SFFT[ipm,:] = np.fft.rfft( SimpleStack[pulse][chan][ipm,:] )
                
                # convert to dB and add colorbars
                #db = 20.*np.log10(np.abs(SFFT[:,istart:iend]))
                db = (np.abs(SFFT[:,istart:iend]))
                
                #db = (np.real(SFFT[:,istart:iend]))
                #db = (np.imag(SFFT[:,istart:iend]))
                #dbr = (np.real(SFFT[:,istart:iend]))
                #db = (np.imag(SFFT[:,istart:iend]))
                
                vvmin =  min(vvmin, np.min(db) + 1e-16 )
                vvmax =  max(vvmax, np.max(db) + 1e-16 )
                im2.append(ax2.matshow( db, aspect='auto', cmap=cmocean.cm.ice, vmin=vvmin, vmax=vvmax))
                #im1.append(ax1.matshow( dbr, aspect='auto')) #, vmin=vvmin, vmax=vvmax))
                #im2.append(ax2.matshow( db, aspect='auto', vmin=vvmin, vmax=vvmax))
                #im2 = ax2.matshow( db, aspect='auto', cmap=cmocean.cm.ice, vmin=vvmin, vmax=vvmax)
                if ichan == 0:
                    #ax2.set_ylabel(r"$q$ (A $\cdot$ s)", fontsize=8)
                    ax2.set_ylabel(r"pulse index", fontsize=10)
                    #ax1.set_ylabel(r"FID (nV)", fontsize=8)
                else:
                    #ax2.yaxis.set_ticklabels([])
                    plt.setp(ax2.get_yticklabels(), visible=False)

                ax2.xaxis.set_major_formatter(formatter)
                ax2.xaxis.set_ticks_position('bottom')
                ax2.xaxis.set_major_locator(MaxNLocator(3))
                
                y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
                ax2.yaxis.set_major_formatter(y_formatter)


                #if chan == self.DATADICT[pulse]["chan"][-1]:
                    #cb2 = canvas.fig.colorbar(im2, cax=cbar_ax, format='%1.0e')

                #cb2 = canvas.fig.colorbar(im2[0], ax=ax2, format='%1.0e', orientation='horizontal')
                #cb2 = canvas.fig.colorbar(im2, ax=ax2, format='%1.0e', orientation='horizontal')
                #cb2.ax.tick_params(axis='both', which='major', labelsize=8)
                #cb2.set_label("signal (dB)", fontsize=8)
 
                ichan += 1   


        canvas.fig.subplots_adjust(hspace=.35, wspace=.15, left=.15, right=.8 )#left=None, bottom=None, right=None, top=None, wspace=None, hspace=None) 
        deSpine(ax1)
        
        #cb1 = canvas.fig.colorbar(im, ax=axes[0::2], format='%1.0e', orientation='horizontal', shrink=.35, aspect=30)
        #cb1.ax.tick_params(axis='both', which='major', labelsize=8)
        #cb1.set_label("$\mathcal{V}_N$ (nV)", fontsize=8)

        cb2 = canvas.fig.colorbar(im2[-1], ax=axes[1::2], format='%1.0e', orientation='horizontal', shrink=.35, aspect=30)
        cb2.ax.tick_params(axis='both', which='major', labelsize=fs)
        cb2.set_label(r"$\left| \mathcal{V}_N \right|$ (nV)", fontsize=fs)
                    

        #canvas.fig.tight_layout() 
        canvas.draw()
        self.doneTrigger.emit() 
    
    def harmonicModel(self, nF, \
        f0, f0K1, f0KN, f0Ks, f0ns, \
        f1, f1K1, f1KN, f1Ks,  \
        Nsearch, Bounds, procRefs, \
        plot, canvas):
        """ nF = number of base frequencies, must be 1 or 2 
            f0 = first base frequency  
            f0K1 = first harmonic to model for first base frequency 
            f0KN = last harmonic to model for the first base frequency 
            f0Ks = subharmonic spacing, set to 1 for no subharmonics.
            f0Ns = number of segments for f0
            f1 = second base frequency  
            f1K1 = first harmonic to model for second base frequency 
            f1KN = last harmonic to model for the second base frequency 
            f1Ks = subharmonic spacing for the second base frequency, set to 1 for no subharmonics.
            Nsearch = the number of harmonics to use when determining base frequency 
            bounds = 1/2 the width of the space where baseline frequency will be searched  
            procRefs = should the reference loops be processed as well  
            plot = should Akvo plot the results 
            canvas = mpl plotting axis      
        """
        TDPlot = True
        fs = 10       
 
        if plot:
            canvas.reAx2(shy=False)
            canvas.ax1.set_ylabel(r"signal (nV)", fontsize=fs)
            canvas.ax2.set_ylabel(r"signal (nV)", fontsize=fs)
            if TDPlot:
                canvas.ax2.set_xlabel(r"time (s)", fontsize=fs)
            else:
                canvas.ax2.set_xlabel(r"frequency (Hz)", fontsize=fs)
                canvas.ax1.set_yscale('log')
                canvas.ax2.set_yscale('log')



        # Data
        iFID = 0


        # stores previous f0 as starting point in non-linear search 
        f0p = {} 
        f1p = {} 
        for pulse in self.DATADICT["PULSES"]:
            for rchan in self.DATADICT[pulse]["rchan"]:
                f0p[rchan] = f0
                f1p[rchan] = f1+1e-1 
            for chan in self.DATADICT[pulse]["chan"]:
                f0p[chan] = f0
                f1p[chan] = f1+1e-1

        for pulse in self.DATADICT["PULSES"]:
            Nseg = int( np.floor(len( self.DATADICT[pulse]["TIMES"] ) / f0ns) )
            for istack in self.DATADICT["stacks"]:
                for ipm in range(self.DATADICT["nPulseMoments"]):
                    if plot:
                        canvas.softClear()
                        mmaxr = 0
                        mmaxd = 0
                        if procRefs: 
                            for ichan in self.DATADICT[pulse]["rchan"]:
                                if TDPlot:
                                    canvas.ax1.plot( self.DATADICT[pulse]["TIMES"], 1e9*self.DATADICT[pulse][ichan][ipm][istack], alpha=.5) 
                                    mmaxr = max( mmaxr, np.max(1e9*self.DATADICT[pulse][ichan][ipm][istack])) 
                                else:
                                    ww = np.fft.fftfreq(len(self.DATADICT[pulse][ichan][ipm][istack]), d=self.dt)
                                    X = np.fft.rfft(self.DATADICT[pulse][ichan][ipm][istack])
                                    canvas.ax1.plot(np.abs(ww[0:len(X)]), np.abs(X), alpha=.5)
                            canvas.ax1.set_prop_cycle(None)
                            canvas.ax1.set_ylim(-mmaxr, mmaxr) 
                        for ichan in self.DATADICT[pulse]["chan"]:
                            if TDPlot:
                                canvas.ax2.plot( self.DATADICT[pulse]["TIMES"], 1e9*self.DATADICT[pulse][ichan][ipm][istack], alpha=.5) 
                                mmaxd = max( mmaxd, np.max(1e9*self.DATADICT[pulse][ichan][ipm][istack])) 
                            else:
                                ww = np.fft.fftfreq(len(self.DATADICT[pulse][ichan][ipm][istack]), d=self.dt)
                                X = np.fft.rfft(self.DATADICT[pulse][ichan][ipm][istack])
                                canvas.ax2.plot(np.abs(ww[0:len(X)]), np.abs(X), alpha=.5)
                        canvas.ax2.set_prop_cycle(None)
                        canvas.ax2.set_ylim(-mmaxd, mmaxd)
                    if procRefs: 
                        for ichan in self.DATADICT[pulse]["rchan"]:
                            if nF == 1:
                                for iseg in range(f0ns):
                                    if iseg < f0ns-1:
                                        self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg:(iseg+1)*Nseg], f0p[ichan] = \
                                            harmonic.minHarmonic( self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg:(iseg+1)*Nseg], \
                                            self.samp,  self.DATADICT[pulse]["TIMES"][iseg*Nseg:(iseg+1)*Nseg], \
                                            f0p[ichan], f0K1, f0KN, f0Ks, Bounds, Nsearch ) 
                                    else:
                                        self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg::], f0p[ichan] = \
                                            harmonic.minHarmonic( self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg::], \
                                            self.samp,  self.DATADICT[pulse]["TIMES"][iseg*Nseg::], \
                                            f0p[ichan], f0K1, f0KN, f0Ks, Bounds, Nsearch ) 
                            elif nF == 2:
                                for iseg in range(f0ns):
                                    if iseg < f0ns-1:
                                        self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg:(iseg+1)*Nseg], f0p[ichan], f1p[ichan] = \
                                            harmonic.minHarmonic2( self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg:(iseg+1)*Nseg],\
                                        self.samp,  self.DATADICT[pulse]["TIMES"][iseg*Nseg:(iseg+1)*Nseg], \
                                        f0p[ichan], f0K1, f0KN, f0Ks,  \
                                        f1p[ichan], f1K1, f1KN, f1Ks, Bounds, Nsearch ) 
                                    else:
                                        self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg::], f0p[ichan], f1p[ichan] = \
                                            harmonic.minHarmonic2( self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg::],\
                                        self.samp,  self.DATADICT[pulse]["TIMES"][iseg*Nseg::], \
                                        f0p[ichan], f0K1, f0KN, f0Ks,  \
                                        f1p[ichan], f1K1, f1KN, f1Ks, Bounds, Nsearch ) 
                            # plot
                            if plot:
                                if TDPlot:
                                    canvas.ax1.plot( self.DATADICT[pulse]["TIMES"], 1e9*self.DATADICT[pulse][ichan][ipm][istack], \
                                        label = pulse + " ipm=" + str(ipm) + " istack=" + str(istack) + " rchan="  + str(ichan))
                                else:
                                    ww = np.fft.fftfreq(len(self.DATADICT[pulse][ichan][ipm][istack]), d=self.dt)
                                    X = np.fft.rfft(self.DATADICT[pulse][ichan][ipm][istack])
                                    canvas.ax1.plot(np.abs(ww[0:len(X)]), np.abs(X),\
                                    label = pulse + " ipm=" + str(ipm) + " istack=" + str(istack) + " rchan="  + str(ichan))

                    for ichan in self.DATADICT[pulse]["chan"]:
                        if nF == 1:
                            for iseg in range(f0ns):
                                if iseg < f0ns-1:
                                    self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg:(iseg+1)*Nseg], f0p[ichan] = \
                                        harmonic.minHarmonic( self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg:(iseg+1)*Nseg], 
                                            self.samp,  self.DATADICT[pulse]["TIMES"][iseg*Nseg:(iseg+1)*Nseg], \
                                            f0p[ichan], f0K1, f0KN, f0Ks, Bounds, Nsearch ) 
                                else:
                                    self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg::], f0p[ichan] = \
                                        harmonic.minHarmonic( self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg::], 
                                            self.samp,  self.DATADICT[pulse]["TIMES"][iseg*Nseg::], \
                                            f0p[ichan], f0K1, f0KN, f0Ks, Bounds, Nsearch )

                        elif nF == 2:
                            for iseg in range(f0ns):
                                if iseg < f0ns-1:
                                    self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg:(iseg+1)*Nseg], f0p[ichan], f1p[ichan] = \
                                        harmonic.minHarmonic2( self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg:(iseg+1)*Nseg],\
                                     self.samp,  self.DATADICT[pulse]["TIMES"][iseg*Nseg:(iseg+1)*Nseg], \
                                     f0p[ichan], f0K1, f0KN, f0Ks,  \
                                     f1p[ichan], f1K1, f1KN, f1Ks, Bounds, Nsearch ) 
                                else:
                                    self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg::], f0p[ichan], f1p[ichan] = \
                                        harmonic.minHarmonic2( self.DATADICT[pulse][ichan][ipm][istack][iseg*Nseg::],\
                                     self.samp,  self.DATADICT[pulse]["TIMES"][iseg*Nseg::], \
                                     f0p[ichan], f0K1, f0KN, f0Ks,  \
                                     f1p[ichan], f1K1, f1KN, f1Ks, Bounds, Nsearch ) 
               
                        # plot
                        if plot:
                            if TDPlot:
                                canvas.ax2.plot( self.DATADICT[pulse]["TIMES"], 1e9*self.DATADICT[pulse][ichan][ipm][istack], \
                                    label = pulse + " ipm=" + str(ipm) + " istack=" + str(istack) + " chan="  + str(ichan))
                            else:
                                ww = np.fft.fftfreq(len(self.DATADICT[pulse][ichan][ipm][istack]), d=self.dt)
                                X = np.fft.rfft(self.DATADICT[pulse][ichan][ipm][istack])
                                canvas.ax2.plot(np.abs(ww[0:len(X)]), np.abs(X), \
                                    label = pulse + " ipm=" + str(ipm) + " istack=" + str(istack) + " chan="  + str(ichan))

                    if plot:
                        if procRefs: 
                            canvas.ax1.legend(prop={'size':fs}, loc='upper right')
                            plt.setp(canvas.ax1.get_xticklabels(), visible=False)
                        canvas.ax2.legend(prop={'size':fs}, loc='upper right')
                        deSpine(canvas.ax1)
                        deSpine(canvas.ax2)
                        canvas.fig.tight_layout()
                        canvas.draw() 
                
                percent = (int)(1e2*((ipm+istack*self.nPulseMoments)/(self.nPulseMoments*len(self.DATADICT["stacks"])))) 
                self.progressTrigger.emit(percent)  
            iFID += 1

        self.doneTrigger.emit() 
        self.updateProcTrigger.emit()  
        self.doneTrigger.emit() 
    
    def FDSmartStack(self, outlierTest, MADcutoff, canvas):
        
        print("FFT stuff")
        self.dataCubeFFT()       

        Stack = {}
        # align phase cycling for stacking and modulate
        for pulse in self.DATADICT["PULSES"]:
            stack = np.zeros(( len(self.DATADICT[pulse]["chan"]), \
                               self.DATADICT["nPulseMoments"],\
                               len(self.DATADICT["stacks"]),\
                               len(self.DATADICT[pulse][self.DATADICT[pulse]["chan"][0] ]["FFT"]["nu"])//2 + 1),\
                               dtype=np.complex )
            for ipm in range(self.DATADICT["nPulseMoments"]):
                istack = 0
                for sstack in self.DATADICT["stacks"]:
                    if self.pulseType == "FID" or pulse == "Pulse 2":
                        mod = (-1)**(ipm%2) * (-1)**(sstack%2)
                    elif self.pulseType == "4PhaseT1":
                        mod = (-1)**(ipm%2) * (-1)**(((sstack-1)/2)%2)
                    ichan = 0
                    for chan in self.DATADICT[pulse]["chan"]:
                        #stack[ichan,ipm,istack,:] += mod*self.DATADICT[pulse][chan][ipm][sstack]
                        stack[ichan,ipm,istack,:] += mod*self.DATADICT[pulse][chan]["FFT"][sstack][ipm,:] 
                        ichan += 1
                    istack += 1
            Stack[pulse] = stack

        ######################################### 
        # simple stack and plot of simple stack #
        ########################################https://faculty.apps.utah.edu/#
        canvas.reAxH2(np.shape(stack)[0], False, False)
        axes = canvas.fig.axes
        SimpleStack = {}
        VarStack = {}
        for pulse in self.DATADICT["PULSES"]:
            SimpleStack[pulse] = {}
            VarStack[pulse] = {}
            ichan = 0
            for chan in self.DATADICT[pulse]["chan"]: 
                SimpleStack[pulse][chan] = 1e9*np.average( Stack[pulse][ichan], 1 ) 
                VarStack[pulse][chan] = 1e9*np.std( Stack[pulse][ichan], 1 ) 
                ax1 = axes[ 2*ichan ]
                #ax1.get_yaxis().get_major_formatter().set_useOffset(False)

                y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
                ax1.yaxis.set_major_formatter(y_formatter)

                #ax1.plot( 1e3*self.DATADICT[pulse][chan]["FFT"]["nu"][0:len(SimpleStack[pulse][chan])], np.average(SimpleStack[pulse][chan], 0 )) #, color='darkblue' )
                #ax1.pcolor( np.real(SimpleStack[pulse][chan]) ) #, color='darkblue' )
                ax1.matshow( np.real(SimpleStack[pulse][chan]), aspect='auto') #, color='darkblue' )
                ax1.set_title("Ch." + str(chan) + ": avg FID", fontsize=10)
                ax1.set_xlabel(r"time (ms)", fontsize=10)

                if ichan == 0:
                    ax1.set_ylabel(r"signal [nV]", fontsize=10)
                else:
                    plt.setp(ax1.get_yticklabels(), visible=False)
                    plt.setp(ax1.get_yaxis().get_offset_text(), visible=False) 
                ichan += 1

        #########################
        # Oulier rejectig stack #
        #########################
        if outlierTest == "MAD":
            MADStack = {}
            VarStack = {}
            #1.4826 is assumption of gaussian noise 
            madstack = np.zeros(( len(self.DATADICT[pulse]["chan"]),\
                                  self.DATADICT["nPulseMoments"],\
                                  len(self.DATADICT[pulse][self.DATADICT[pulse]["chan"][0] ]["FFT"]["nu"])//2 + 1))
            varstack = np.zeros(( len(self.DATADICT[pulse]["chan"]),\
                                  self.DATADICT["nPulseMoments"],\
                                  len(self.DATADICT[pulse][self.DATADICT[pulse]["chan"][0] ]["FFT"]["nu"])//2 + 1))
            for pulse in self.DATADICT["PULSES"]:
                MADStack[pulse] = {}
                VarStack[pulse] = {}
                ichan = 0
                for chan in self.DATADICT[pulse]["chan"]:
                    ax1 = axes[ 2*ichan  ]
                    for ipm in range(self.DATADICT["nPulseMoments"]):
#                         # brutal loop over time, can this be vectorized? 
#                         for it in range(len(self.DATADICT[pulse]["TIMES"])): 
#                             x = 1e9 *Stack[pulse][ichan,ipm,:,it]
#                             MAD = 1.4826 * np.median( np.abs(x-np.median(x)) )
#                             good = 0
#                             for istack in self.DATADICT["stacks"]:
#                                 if (np.abs(x[istack-1]-np.median(x))) / MAD < 2:
#                                     good += 1
#                                     madstack[ ichan, ipm, it ] += x[istack-1]
#                                 else:
#                                     pass
#                             madstack[ichan, ipm, it] /= good
#                         percent = int(1e2* (float)(ipm) / (float)(self.DATADICT["nPulseMoments"]) )
#                         self.progressTrigger.emit(percent)

                        # Vectorized version of above...much, much faster 
                        x = 1e9*copy.deepcopy(Stack[pulse][ichan][ipm,:,:])      # stack and time indices
                        tile_med =  np.tile( np.median(x, axis=0), (np.shape(x)[0],1)) 
                        MAD = MADcutoff * np.median(np.abs(x - tile_med), axis=0)
                        tile_MAD =  np.tile( MAD, (np.shape(x)[0],1)) 
                        good = np.abs(x-tile_med)/tile_MAD < 2. # 1.4826 # 2

                        madstack[ichan][ipm] = copy.deepcopy( np.ma.masked_array(x, good != True).mean(axis=0) )
                        varstack[ichan][ipm] = copy.deepcopy( np.ma.masked_array(x, good != True).std(axis=0) )
                        
                        # reporting
                        percent = int(1e2* (float)((ipm)+ichan*self.DATADICT["nPulseMoments"]) / 
                                           (float)(self.DATADICT["nPulseMoments"] * len(self.DATADICT[pulse]["chan"])))
                        self.progressTrigger.emit(percent)

                    ax2 = axes[2*ichan+1] # TODO fix hard coded number
                    #ax1.plot( 1e3*self.DATADICT[pulse]["TIMES"], np.average(  madstack[ichan], 0  ))# , color='darkred')
                    MADStack[pulse][chan] = madstack[ichan]
                    VarStack[pulse][chan] = varstack[ichan]
                    ax2.matshow( np.real(MADStack[pulse][chan]), aspect='auto') #, color='darkblue' )
                    ichan += 1
 
            self.DATADICT["stack"] = MADStack 

        else:
            self.DATADICT["stack"] = SimpleStack 
 
#         #########################################
#         # Plot Fourier Transform representation #
#         #########################################
# 
# #         canvas.fig.subplots_adjust(right=0.8)
# #         cbar_ax = canvas.fig.add_axes([0.85, 0.1, 0.015, 0.355])
# #         cbar_ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')                 
#         im2 = []
#         im1 = []
#         for pulse in self.DATADICT["PULSES"]:
#             ichan = 0
#             axes = canvas.fig.axes
#             vvmin = 1e10 
#             vvmax = 0
#             for chan in self.DATADICT[pulse]["chan"]: 
#                 ax1 = axes[2*ichan  ]
#                 ax2 = axes[2*ichan+1] # TODO fix hard coded number
#                 if outlierTest == "MAD":
#                     X = np.fft.rfft( MADStack[pulse][chan][0,:] )
#                     nu = np.fft.fftfreq(len( MADStack[pulse][chan][0,:]), d=self.dt)
#                 else:
#                     X = np.fft.rfft( SimpleStack[pulse][chan][0,:] )
#                     nu = np.fft.fftfreq(len( SimpleStack[pulse][chan][0,:]), d=self.dt)
#                 
#                 nu = nu[0:len(X)]
#                 nu[-1] = np.abs(nu[-1])
#                 df = nu[1] - nu[0]
#                 of = 0
# 
#                 istart = int((self.transFreq-50.)/df)
#                 iend = int((self.transFreq+50.)/df)
#                 of = nu[istart]
#                 
#                 def freqlabel(xxx, pos):
#                     return  '%1.0f' %(of + xxx*df)
#                 formatter = FuncFormatter(freqlabel)
#         
#                 SFFT = np.zeros( (self.DATADICT["nPulseMoments"], len(X)), dtype=np.complex64 )
#                 SFFT[0,:] = X
#                 for ipm in range(1, self.DATADICT["nPulseMoments"]):
#                     if outlierTest == "MAD":
#                         SFFT[ipm,:] = np.fft.rfft( MADStack[pulse][chan][ipm,:] )
#                     else:
#                         SFFT[ipm,:] = np.fft.rfft( SimpleStack[pulse][chan][ipm,:] )
#                 
#                 # convert to dB and add colorbars
#                 #db = 20.*np.log10(np.abs(SFFT[:,istart:iend]))
#                 db = (np.abs(SFFT[:,istart:iend]))
#                 #db = (np.real(SFFT[:,istart:iend]))
#                 #dbr = (np.real(SFFT[:,istart:iend]))
#                 #db = (np.imag(SFFT[:,istart:iend]))
#                 
#                 vvmin =  min(vvmin, np.min (db))
#                 vvmax =  max(vvmax, np.max (db))
#                 im2.append(ax2.matshow( db, aspect='auto', cmap=cmocean.cm.ice, vmin=vvmin, vmax=vvmax))
#                 #im1.append(ax1.matshow( dbr, aspect='auto')) #, vmin=vvmin, vmax=vvmax))
#                 #im2.append(ax2.matshow( db, aspect='auto', vmin=vvmin, vmax=vvmax))
#                 #im2 = ax2.matshow( db, aspect='auto', cmap=cmocean.cm.ice, vmin=vvmin, vmax=vvmax)
#                 if ichan == 0:
#                     ax2.set_ylabel(r"$q$ (A $\cdot$ s)", fontsize=8)
#                 else:
#                     #ax2.yaxis.set_ticklabels([])
#                     plt.setp(ax2.get_yticklabels(), visible=False)
# 
#                 ax2.xaxis.set_major_formatter(formatter)
#                 ax2.xaxis.set_ticks_position('bottom')
#                 ax2.xaxis.set_major_locator(MaxNLocator(3))
#                 
#                 y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
#                 ax2.yaxis.set_major_formatter(y_formatter)
# 
# 
#                 #if chan == self.DATADICT[pulse]["chan"][-1]:
#                     #cb2 = canvas.fig.colorbar(im2, cax=cbar_ax, format='%1.0e')
# 
#                 #cb2 = canvas.fig.colorbar(im2[0], ax=ax2, format='%1.0e', orientation='horizontal')
#                 #cb2 = canvas.fig.colorbar(im2, ax=ax2, format='%1.0e', orientation='horizontal')
#                 #cb2.ax.tick_params(axis='both', which='major', labelsize=8)
#                 #cb2.set_label("signal (dB)", fontsize=8)
#  
#                 ichan += 1   
# 
# 
#         canvas.fig.subplots_adjust(hspace=.1, wspace=.05, left=.075, right=.95 )#left=None, bottom=None, right=None, top=None, wspace=None, hspace=None) 
#         
#         #cb1 = canvas.fig.colorbar(im, ax=axes[0::2], format='%1.0e', orientation='horizontal', shrink=.35, aspect=30)
#         #cb1.ax.tick_params(axis='both', which='major', labelsize=8)
#         #cb1.set_label("$\mathcal{V}_N$ (nV)", fontsize=8)
# 
#         cb2 = canvas.fig.colorbar(im2[-1], ax=axes[1::2], format='%1.0e', orientation='horizontal', shrink=.35, aspect=30)
#         cb2.ax.tick_params(axis='both', which='major', labelsize=8)
#         cb2.set_label("$\mathcal{V}_N$ (nV)", fontsize=8)

        #canvas.fig.tight_layout() 
        deSpine(ax1)
        canvas.draw()
        self.doneTrigger.emit() 


    def sumData(self, canvas, fred):
        chans = copy.deepcopy(self.DATADICT[self.DATADICT["PULSES"][0]]["chan"]) #= np.array( ( self.DATADICT[pulse]["chan"][0], ) )
        nchan = len(chans)
        # Sum permutations of two channel combos
        for ich in range(nchan-1):
            for ch in chans[ich+1:]:
                chsum = chans[ich] + "+" + ch
                for pulse in self.DATADICT["PULSES"]:
                    self.DATADICT[pulse][chsum] = {} 
                    for ipm in range(self.DATADICT["nPulseMoments"]):
                        self.DATADICT[pulse][chsum][ipm] = {} 
                        for istack in self.DATADICT["stacks"]:
                            self.DATADICT[pulse][chsum][ipm][istack] = self.DATADICT[pulse][chans[ich]][ipm][istack] - self.DATADICT[pulse][ch][ipm][istack] 
                    if chsum == "1+2":
                        #self.DATADICT[pulse]["rchan"].pop()
                        #self.DATADICT[pulse]["rchan"].pop()
                        self.DATADICT[pulse]["chan"].append(chsum)

        # Sum all channels 
        sumall = False
        if sumall:
            chsum = ""
            for ch in chans:
                chsum += ch + "+" 
            chsum = chsum[0:-1] # remove last "+"
         
            for pulse in self.DATADICT["PULSES"]:
                self.DATADICT[pulse][chsum] = {} 
                for ipm in range(self.DATADICT["nPulseMoments"]):
                    self.DATADICT[pulse][chsum][ipm] = {} 
                    for istack in self.DATADICT["stacks"]:
                        self.DATADICT[pulse][chsum][ipm][istack] = copy.deepcopy(self.DATADICT[pulse][chans[0]][ipm][istack])
                        for ch in chans[1:]:
                            self.DATADICT[pulse][chsum][ipm][istack] += self.DATADICT[pulse][ch][ipm][istack] 
            self.DATADICT[pulse]["chan"].append(chsum)

#         if nchan > 2:
#             for ch in chans:
#                 chsum += ch
#             for ch2 in chans[1::]:
#                 for pulse in self.DATADICT["PULSES"]:
#                     self.DATADICT[pulse][chsum] = {} 
#                         for istack in self.DATADICT["stacks"]:
#                             self.DATADICT[pulse][chsum][ipm][istack] = self.DATADICT[pulse][chans[ich]][ipm][istack] + self.DATADICT[pulse][ch][ipm][istack] 
        self.doneTrigger.emit() 

    def quadDet(self, clip, method, loss, canvas):

        from scipy import signal
        self.RotatedAmplitude = True 

        wL = self.transFreq * 2*np.pi       
        vL = self.transFreq
        #T = 50
        dt = self.dt 
        #DT = 0.01

        CA = {} # corrected amplitude        
        IP = {} # instantaneous phase
        NR = {} # Noise residual
        RE = {} # Real channel 
        IM = {} # Imaginary channel 

        # global maximums for plotting 
        CAmax = {} 
        NRmax = {}
        REmax = {}
        IMmax = {}
        E0,phi,df,T2 = 100.,0,0,.2
        first = False
        self.sigma = {}
        for pulse in self.DATADICT["PULSES"]:
            CA[pulse] = {}
            IP[pulse] = {}
            NR[pulse] = {}
            RE[pulse] = {}
            IM[pulse] = {}
            CAmax[pulse] = 0
            NRmax[pulse] = 0
            REmax[pulse] = 0
            IMmax[pulse] = 0
            ichan = 0
            self.sigma[pulse] = {}
            for chan in self.DATADICT[pulse]["chan"]: 
                CA[pulse][chan] = np.zeros( (self.DATADICT["nPulseMoments"],  len(self.DATADICT[pulse]["TIMES"])-clip )  )
                IP[pulse][chan] = np.zeros( (self.DATADICT["nPulseMoments"],  len(self.DATADICT[pulse]["TIMES"])-clip )  )
                NR[pulse][chan] = np.zeros( (self.DATADICT["nPulseMoments"],  len(self.DATADICT[pulse]["TIMES"])-clip )  )
                RE[pulse][chan] = np.zeros( (self.DATADICT["nPulseMoments"],  len(self.DATADICT[pulse]["TIMES"])-clip )  )
                IM[pulse][chan] = np.zeros( (self.DATADICT["nPulseMoments"],  len(self.DATADICT[pulse]["TIMES"])-clip )  )
        
                #QQ = np.average(self.DATADICT[pulse]["Q"], axis=1 )
                #for ipm in np.argsort(QQ):
                for ipm in range(0, self.DATADICT["nPulseMoments"]):
                    #t = self.DATADICT[pulse]["TIMES"] - self.DATADICT[pulse]["PULSE_TIMES"][-1]
                    xn = self.DATADICT["stack"][pulse][chan][ipm,:]
                    ht = signal.hilbert(xn)*np.exp(-1j*wL*self.DATADICT[pulse]["TIMES"])
                    #############################################################
                    # Quadrature signal 
                    RE[pulse][chan][ipm,:] =  np.real(ht[clip::])  # *-1 for negative for consistency with VC ??
                    IM[pulse][chan][ipm,:] =  np.imag(ht[clip::])
                    REmax[pulse] = max(REmax[pulse], np.max(np.real(ht[clip::])))
                    IMmax[pulse] = max(IMmax[pulse], np.max(np.imag(ht[clip::])))
                    #############################################################
                    # Instantaneous phase 
                    IP[pulse][chan][ipm,:] = np.angle(ht)[clip::]
                    #############################################################
                    # Rotated amplitude
                    #if ipm != 0:
                    #    [success, E0, df, phi, T2] = decay.quadratureDetect2( ht.real, ht.imag, self.DATADICT[pulse]["TIMES"], (E0,phi,df,T2))
                    #[success, E0, df, phi, T2] = decay.quadratureDetect( ht.real, ht.imag, self.DATADICT[pulse]["TIMES"] )
                    #else:
                    [success, E0, df, phi, T2] = decay.quadratureDetect2( ht.real, ht.imag, self.DATADICT[pulse]["TIMES"], method, loss)
                    #[success, E0, df, phi, T2] = decay.quadratureDetect2( ht.real, ht.imag, self.DATADICT[pulse]["TIMES"], (E0,phi,df,T2))
                    #[success, E0, df, phi, T2] = decay.quadratureDetect( ht.real, ht.imag, self.DATADICT[pulse]["TIMES"] )
                    #print("success", success, "E0", E0, "phi", phi, "df", df, "T2", T2)
                    
                    D = self.RotateAmplitude( ht.real, ht.imag, phi, df, self.DATADICT[pulse]["TIMES"] )
                    CA[pulse][chan][ipm,:] = D.imag[clip::]  # amplitude data 
                    NR[pulse][chan][ipm,:] = D.real[clip::]  # noise data
                    CAmax[pulse] = max(CAmax[pulse], np.max(D.imag[clip::]) )
                    NRmax[pulse] = max(NRmax[pulse], np.max(D.real[clip::]) )
                    self.sigma[pulse][chan] = np.std(NR[pulse][chan])
                    # reporting
                    percent = int(1e2* (float)((ipm)+ichan*self.DATADICT["nPulseMoments"]) / 
                                       (float)(self.DATADICT["nPulseMoments"] * len(self.DATADICT[pulse]["chan"])))
                    self.progressTrigger.emit(percent)
                ichan += 1
            self.DATADICT[pulse]["TIMES"] = self.DATADICT[pulse]["TIMES"][clip::]
            
        self.DATADICT["CA"] = CA
        self.DATADICT["IP"] = IP
        self.DATADICT["NR"] = NR
        self.DATADICT["RE"] = RE
        self.DATADICT["IM"] = IM
        
        self.DATADICT["CAmax"] = CAmax
        self.DATADICT["NRmax"] = NRmax
        self.DATADICT["REmax"] = REmax
        self.DATADICT["IMmax"] = IMmax
        
        self.doneTrigger.emit() 

    def plotQuadDet(self, clip, phase, canvas):
        
        canvas.reAxH2(  len(self.DATADICT[ self.DATADICT["PULSES"][0] ]["chan"] ), False, False)

        ###############
        # Plot on GUI #      
        ###############
        fs = 10 
        dcmap = cmocean.cm.curl_r  #"seismic_r" #cmocean.cm.balance_r #"RdBu" #YlGn" # "coolwarm_r"  # diverging 
        canvas.reAxH2(  len(self.DATADICT[ self.DATADICT["PULSES"][0] ]["chan"] ), False, False)
        for pulse in self.DATADICT["PULSES"]:
            ichan = 0
            axes = canvas.fig.axes
            mmaxr = 0.
            mmaxi = 0.
            #if clip > 0:
            #    time_sp =  1e3 * (self.DATADICT[pulse]["TIMES"][clip-1::] - self.DATADICT[pulse]["PULSE_TIMES"][-1] )
            #else:
            #    time_sp =  1e3 * (self.DATADICT[pulse]["TIMES"] - self.DATADICT[pulse]["PULSE_TIMES"][-1] )
            time_sp =  1e3 * (self.DATADICT[pulse]["TIMES"] - self.DATADICT[pulse]["PULSE_TIMES"][-1] )
                
            QQ = np.average(self.DATADICT[pulse]["Q"], axis=1 )
            iQ = np.argsort(QQ)

            for chan in self.DATADICT[pulse]["chan"]: 
                ax1 = axes[2*ichan  ]
                ax2 = axes[2*ichan+1] 
                if phase == 0: # Re Im 
                    im1 = ax1.pcolormesh( time_sp, QQ[iQ], self.DATADICT["RE"][pulse][chan][iQ], cmap=dcmap, \
                         vmin=-self.DATADICT["REmax"][pulse] , vmax=self.DATADICT["REmax"][pulse] , rasterized=True)
                    im2 = ax2.pcolormesh( time_sp, QQ[iQ], self.DATADICT["IM"][pulse][chan][iQ], cmap=dcmap, \
                         vmin=-self.DATADICT["IMmax"][pulse], vmax=self.DATADICT["IMmax"][pulse] , rasterized=True )
                    #im1 = ax1.matshow( self.DATADICT["RE"][pulse][chan][iQ], cmap=dcmap, aspect='auto', \
                    #     vmin=-self.DATADICT["REmax"][pulse] , vmax=self.DATADICT["REmax"][pulse] )
                    #im2 = ax2.matshow( self.DATADICT["IM"][pulse][chan][iQ], cmap=dcmap, aspect='auto', \
                    #     vmin=-self.DATADICT["REmax"][pulse] , vmax=self.DATADICT["REmax"][pulse] )
                if phase == 1: # Amp phase
                    im1 = ax1.pcolormesh( time_sp, QQ[iQ], self.DATADICT["CA"][pulse][chan][iQ], cmap=dcmap, \
                         vmin=-self.DATADICT["CAmax"][pulse] , vmax=self.DATADICT["CAmax"][pulse], rasterized=True  )
                    #im2 = ax2.pcolormesh( time_sp, QQ, self.DATADICT["IP"][pulse][chan], cmap=cmocean.cm.balance, rasterized=True,\
                    im2 = ax2.pcolormesh( time_sp, QQ[iQ], self.DATADICT["IP"][pulse][chan][iQ], cmap=cmocean.cm.delta, \
                         vmin=-np.pi, vmax=np.pi, rasterized=True)
                if phase == 2: # CA NR
                    im1 = ax1.pcolormesh( time_sp, QQ[iQ], self.DATADICT["CA"][pulse][chan][iQ], cmap=dcmap, \
                         vmin=-self.DATADICT["CAmax"][pulse] , vmax=self.DATADICT["CAmax"][pulse], rasterized=True )
                    im2 = ax2.pcolormesh( time_sp, QQ[iQ], self.DATADICT["NR"][pulse][chan][iQ], cmap=dcmap, \
                         vmin=-self.DATADICT["NRmax"][pulse] , vmax=self.DATADICT["NRmax"][pulse], rasterized=True )
#                     cb2 = canvas.fig.colorbar(im2, ax=ax2, format='%1.0e')
#                     cb2.set_label("Noise residual (nV)", fontsize=8)
#                     cb2.ax.tick_params(axis='both', which='major', labelsize=8)
#                     cb1 = canvas.fig.colorbar(im1, ax=ax1, format='%1.0e')
#                     cb1.set_label("Phased amplitude (nV)", fontsize=8)
#                     cb1.ax.tick_params(axis='both', which='major', labelsize=8)

#                     cb2 = canvas.fig.colorbar(im2, ax=ax2, format="%1.0e")
#                     cb2.set_label("Phase (rad)", fontsize=8)
#                     cb2.ax.tick_params(axis='both', which='major', labelsize=8)
#                     cb1 = canvas.fig.colorbar(im1, ax=ax1, format="%1.0e")
#                     cb1.set_label("FID (nV)", fontsize=8)
#                     cb1.ax.tick_params(axis='both', which='major', labelsize=8)

                # if you save these as pdf or eps, there are artefacts 
#                 for cbar in [cb1,cb2]:
#                     #cbar.solids.set_rasterized(True)
#                     cbar.solids.set_edgecolor("face")
                    
                # reporting
                percent = int(1e2* (float)(ichan)/len(self.DATADICT[pulse]["chan"])) 
                self.progressTrigger.emit(percent)
                
                if ichan == 0:
                    ax1.set_ylabel(r"$q$ ( $\mathrm{A}\cdot\mathrm{s}$)", fontsize=fs)
                    ax2.set_ylabel(r"$q$ ( $\mathrm{A}\cdot\mathrm{s}$)", fontsize=fs)
                else:
                    #ax2.yaxis.set_ticklabels([])
                    #ax1.yaxis.set_ticklabels([])
                    plt.setp(ax1.get_yticklabels(), visible=False)
                    plt.setp(ax2.get_yticklabels(), visible=False)
                ichan += 1

            ax1.set_yscale('log')
            ax2.set_yscale('log')

            plt.setp(ax1.get_xticklabels(), visible=False)
 
            ax1.set_ylim( np.min(QQ), np.max(QQ) )
            ax2.set_ylim( np.min(QQ), np.max(QQ) )

            ax1.set_xlim( np.min(time_sp), np.max(time_sp) )
            ax2.set_xlim( np.min(time_sp), np.max(time_sp) )

            #ax2.set_xlabel(r"Time since end of pulse  (ms)", fontsize=8)
            ax2.set_xlabel(r"time (ms)", fontsize=fs)
        
        #canvas.fig.subplots_adjust(hspace=.15, wspace=.05, left=.15, right=.85, bottom=.15, top=.9 )#left=None, bottom=None, right=None, top=None, wspace=None, hspace=None) 
        canvas.fig.subplots_adjust(hspace=.15, wspace=.05, left=.15, right=.90, bottom=.15, top=.95 )#left=None, bottom=None, right=None, top=None, wspace=None, hspace=None) 
        
        tick_locator = MaxNLocator(nbins=3)

        cb1 = canvas.fig.colorbar(im1, ax=axes[0::2], format='%1.0f', orientation='vertical')
        #cb1 = canvas.fig.colorbar(im1, ax=axes[0::2], format='%1.0f', orientation='horizontal', shrink=.35, aspect=30, pad=.4)
        cb1.ax.tick_params(axis='both', which='major', labelsize=fs)

        cb1.locator = tick_locator
        cb1.update_ticks()

        tick_locator2 = MaxNLocator(nbins=3)
        cb2 = canvas.fig.colorbar(im2, ax=axes[1::2], format='%1.0f', orientation='vertical')
        #cb2 = canvas.fig.colorbar(im2, ax=axes[1::2], format='%1.0f', orientation='horizontal', shrink=.35, aspect=30, pad=.4)
        cb2.ax.tick_params(axis='both', which='major', labelsize=fs)

        if phase == 0: # Re Im 
            cb1.set_label(r"$\mathrm{Re} \left( \mathcal{V}_N \right)$ (nV)", fontsize=fs)
            cb2.set_label(r"$\mathrm{Im} \left( \mathcal{V}_N \right)$ (nV)", fontsize=fs)
        elif phase == 1: # Amp phase
            cb1.set_label(r"$\left| \mathcal{V}_N \right|$ (nV)", fontsize=fs)
            cb2.set_label(r"$\angle \mathcal{V}_N$", fontsize=fs)
        else:
            cb1.set_label(r"$\left| \mathcal{V}_N \right|$ (nV)", fontsize=fs)
            cb2.set_label(r"noise (nV)", fontsize=fs)


        cb2.locator = tick_locator2
        cb2.update_ticks()

        #canvas.fig.tight_layout()
        canvas.draw()
        self.doneTrigger.emit() 

    def RotateAmplitude(self, X, Y, zeta, df, t):
        V = X + 1j*Y
        return np.abs(V) * np.exp( 1j * ( np.angle(V) - zeta - 2.*np.pi*df*t ) )

    def gateIntegrate( self, gpd, clip, canvas ):
        """ Gate integrate the real, imaginary, phased, and noise residual channels
        """

        self.gated = True 
        self.GATED = {}
        
        for pulse in self.DATADICT["PULSES"]:
            QQ = np.average(self.DATADICT[pulse]["Q"], axis=1 )
            iQ = np.argsort(QQ)
            ichan = 0
            for chan in self.DATADICT[pulse]["chan"]:
                self.GATED[chan] = {}
                for ipm in range(0, self.DATADICT["nPulseMoments"]):
                #for ipm in iQ: 
                    # Time since pulse end rather than since record starts...
                    #if clip > 0:
                    #    time_sp =  1e3 * (self.DATADICT[pulse]["TIMES"][clip:] - self.DATADICT[pulse]["PULSE_TIMES"][-1] )
                    #else:
                    time_sp =  1e3 * (self.DATADICT[pulse]["TIMES"] - self.DATADICT[pulse]["PULSE_TIMES"][-1] )

                    #GT, GD, GTT, sig_stack, isum      = rotate.gateIntegrate( self.DATADICT["CA"][pulse][chan][ipm,:], time_sp, gpd, self.sigma[pulse][chan], 1.5 )
                    #GT2, GP, GTT, sig_stack_err, isum = rotate.gateIntegrate( self.DATADICT["NR"][pulse][chan][ipm,:], time_sp, gpd, self.sigma[pulse][chan], 1.5 ) 
                   
                    #              err  
                    GT, GCA, GTT, sig_stack, isum  = rotate.gateIntegrate( self.DATADICT["CA"][pulse][chan][ipm], time_sp, gpd, self.sigma[pulse][chan], 2 )
                    GT, GNR, GTT, sig_stack, isum  = rotate.gateIntegrate( self.DATADICT["NR"][pulse][chan][ipm], time_sp, gpd, self.sigma[pulse][chan], 2 )
                    GT, GRE, GTT, sig_stack, isum  = rotate.gateIntegrate( self.DATADICT["RE"][pulse][chan][ipm], time_sp, gpd, self.sigma[pulse][chan], 2 )
                    GT, GIM, GTT, sig_stack, isum  = rotate.gateIntegrate( self.DATADICT["IM"][pulse][chan][ipm], time_sp, gpd, self.sigma[pulse][chan], 2 )
                    GT, GIP, GTT, sig_stack, isum  = rotate.gateIntegrate( self.DATADICT["IP"][pulse][chan][ipm], time_sp, gpd, self.sigma[pulse][chan], 2 )
                    
                    #if ipm == iQ[0]:
                    if ipm == 0:
                    #    self.GATED[chan]["DATA"]  = np.zeros( ( self.DATADICT["nPulseMoments"], len(GT)) )
                    #    self.GATED[chan]["ERR"]   = np.zeros( ( self.DATADICT["nPulseMoments"], len(GT)) )
                    #    self.GATED[chan]["SIGMA"] = np.zeros( ( self.DATADICT["nPulseMoments"], len(GT)) )
                        self.GATED[chan]["CA"] = np.zeros( ( self.DATADICT["nPulseMoments"], len(GT)-clip) )
                        self.GATED[chan]["NR"] = np.zeros( ( self.DATADICT["nPulseMoments"], len(GT)-clip) )
                        self.GATED[chan]["BN"] = np.zeros( ( self.DATADICT["nPulseMoments"], len(GT)-clip) )
                        self.GATED[chan]["RE"] = np.zeros( ( self.DATADICT["nPulseMoments"], len(GT)-clip) )
                        self.GATED[chan]["IM"] = np.zeros( ( self.DATADICT["nPulseMoments"], len(GT)-clip) )
                        self.GATED[chan]["IP"] = np.zeros( ( self.DATADICT["nPulseMoments"], len(GT)-clip) )
                        self.GATED[chan]["isum"] = isum
                
                    # Bootstrap noise 
                    #self.GATED[chan]["isum"]
                    Means = rotate.bootstrapWindows( self.DATADICT["NR"][pulse][chan][ipm], 20000, isum[isum!=1], adapt=True)
                    # MAD, only for windows > 1 
                    c = stats.norm.ppf(3./4.)
                    sig_stack[isum!=1] = np.ma.median(np.ma.abs(Means), axis=1) / c 
                    self.GATED[chan]["BN"][ipm] = sig_stack[clip:] 

                    #self.GATED[chan]["DATA"][ipm] = GD.real
                    self.GATEDABSCISSA = GT[clip:]
                    self.GATEDWINDOW = GTT[clip:]
                    #self.GATED[chan]["SIGMA"][ipm] =  sig_stack #_err # GP.real
                    #self.GATED[chan]["ERR"][ipm] =  GP.real
                    
                    #self.GATED[chan]["CA"][iQ[ipm]] = GCA.real[clip:]
                    #self.GATED[chan]["NR"][iQ[ipm]] = GNR.real[clip:]
                    #self.GATED[chan]["RE"][iQ[ipm]] = GRE.real[clip:]
                    #self.GATED[chan]["IM"][iQ[ipm]] = GIM.real[clip:]
                    #self.GATED[chan]["IP"][iQ[ipm]] = GIP.real[clip:]
                    self.GATED[chan]["CA"][ipm] = GCA.real[clip:]
                    self.GATED[chan]["NR"][ipm] = GNR.real[clip:]
                    self.GATED[chan]["RE"][ipm] = GRE.real[clip:]
                    self.GATED[chan]["IM"][ipm] = GIM.real[clip:]
                    self.GATED[chan]["IP"][ipm] = GIP.real[clip:]
                    
                    percent = int(1e2* (float)((ipm)+ichan*self.DATADICT["nPulseMoments"]) / 
                                       (float)(self.DATADICT["nPulseMoments"] * len(self.DATADICT[pulse]["chan"])))
                    self.progressTrigger.emit(percent)


                self.GATED[chan]["CA"] = self.GATED[chan]["CA"][iQ,:]
                self.GATED[chan]["NR"] = self.GATED[chan]["NR"][iQ,:]
                self.GATED[chan]["RE"] = self.GATED[chan]["RE"][iQ,:]
                self.GATED[chan]["IM"] = self.GATED[chan]["IM"][iQ,:]
                self.GATED[chan]["IP"] = self.GATED[chan]["IP"][iQ,:]
                self.GATED[chan]["GTT"] = GTT[clip:]
                self.GATED[chan]["GT"] = GT[clip:]
                self.GATED[chan]["QQ"] = QQ[iQ]
                ichan += 1
        self.doneTrigger.emit() 

    def bootstrap_resample(self, X, n=None):
        # from http://nbviewer.jupyter.org/gist/aflaxman/6871948
        """ Bootstrap resample an array_like
        Parameters
        ----------
        X : array_like
        data to resample
        n : int, optional
        length of resampled array, equal to len(X) if n==None
        Results
        -------
        returns X_resamples
        """
        if n == None:
            n = len(X)
        
        resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
        return X[resample_i]

    def bootstrap_sigma(self, pulse, chan):
                    
        # bootstrap resample
        nt = len(self.GATED[chan]["GT"])
        nb = 5000
        XS = np.zeros( (nb, nt) )
        for ii in range(nb):
            for it in range(nt):
                if self.GATED[chan]["isum"][it] < 8:
                    XS[ii, it] = self.sigma[pulse][chan]
                else:
                    if it == 0:                
                        X = self.bootstrap_resample( np.concatenate( (self.GATED[chan]["NR"][:,it],   \
                                                                      self.GATED[chan]["NR"][:,it+1], \
                                                                      self.GATED[chan]["NR"][:,it+2], \
                                                                      self.GATED[chan]["NR"][:,it+3] ) ), n=nt )
                    elif it == 1:                
                        X = self.bootstrap_resample( np.concatenate( (self.GATED[chan]["NR"][:,it-1], self.GATED[chan]["NR"][:,it], \
                            self.GATED[chan]["NR"][:,it+1], self.GATED[chan]["NR"][:,it+2] ) ), n=nt )
                    elif it ==  nt-2:              
                        X = self.bootstrap_resample( np.concatenate( (self.GATED[chan]["NR"][:,it+1], self.GATED[chan]["NR"][:,it], \
                            self.GATED[chan]["NR"][:,it-1], self.GATED[chan]["NR"][:,it-2] ) ), n=nt )
                    elif it ==  nt-1:              
                        X = self.bootstrap_resample( np.concatenate( (self.GATED[chan]["NR"][:,it], self.GATED[chan]["NR"][:,it-1], \
                            self.GATED[chan]["NR"][:,it-2], self.GATED[chan]["NR"][:,it-3] ) ), n=nt )
                    else:              
                        X = self.bootstrap_resample( np.concatenate( (self.GATED[chan]["NR"][:,it-2] , self.GATED[chan]["NR"][:,it-1], \
                            self.GATED[chan]["NR"][:,it], self.GATED[chan]["NR"][:,it+1], self.GATED[chan]["NR"][:,it+2] )), n=nt )
                    XS[ii,it] = np.std(X)
        return XS

    def plotGateIntegrate( self, gpd, clip, phase, canvas ):
        """ Plot the gate integration 
        """
       
        fs = 10 
 
        canvas.reAxH2(  len(self.DATADICT[ self.DATADICT["PULSES"][0] ]["chan"] ), False, False)
        axes = canvas.fig.axes
        #cmap = cmocean.cm.balance_r 
        dcmap = cmocean.cm.curl_r  #"seismic_r" #cmocean.cm.balance_r #"RdBu" #YlGn" # "coolwarm_r"  # diverging 
    
        # Calculate maximum for plotting...TODO move into loop above
        vmax1 = 0
        vmax2 = 0
        for pulse in self.DATADICT["PULSES"]:
            for chan in self.DATADICT[pulse]["chan"]:
                if phase == 0:
                    vmax1 = max(vmax1, np.max(np.abs(self.GATED[chan]["RE"])))
                    vmax2 = max(vmax2, np.max(np.abs(self.GATED[chan]["IM"])))
                elif phase == 1:
                    vmax1 = max(vmax1, np.max(np.abs(self.GATED[chan]["CA"])))
                    vmax2 = np.pi
                elif phase == 2:
                    vmax1 = max(vmax1, np.max(np.abs(self.GATED[chan]["CA"])))
                    vmax2 = max(vmax2, np.max(np.abs(self.GATED[chan]["NR"])))
        

        for pulse in self.DATADICT["PULSES"]:
            ichan = 0
            for chan in self.DATADICT[pulse]["chan"]:
            
                ax1 = axes[2*ichan  ]
                ax2 = axes[2*ichan+1] 
                
                if phase == 0:
                    im1 = ax1.pcolormesh(self.GATED[chan]["GTT"], self.GATED[chan]["QQ"], self.GATED[chan]["RE"], cmap=dcmap, vmin=-vmax1, vmax=vmax1)
                    im2 = ax2.pcolormesh(self.GATED[chan]["GTT"], self.GATED[chan]["QQ"], self.GATED[chan]["IM"], cmap=dcmap, vmin=-vmax2, vmax=vmax2)
                    #im1 = ax1.matshow(self.GATED[chan]["RE"], cmap=dcmap, vmin=-vmax1, vmax=vmax1, aspect='auto')
                    #im2 = ax2.matshow(self.GATED[chan]["IM"], cmap=dcmap, vmin=-vmax2, vmax=vmax2, aspect='auto')
                elif phase == 1:
                    im1 = ax1.pcolormesh(self.GATED[chan]["GTT"], self.GATED[chan]["QQ"], self.GATED[chan]["CA"], cmap=dcmap, vmin=-vmax1, vmax=vmax1)
                    im2 = ax2.pcolormesh(self.GATED[chan]["GTT"], self.GATED[chan]["QQ"], self.GATED[chan]["IP"], cmap=cmocean.cm.delta, vmin=-vmax2, vmax=vmax2)
                    #im2 = ax2.pcolormesh(self.GATED[chan]["GTT"], self.GATED[chan]["QQ"], self.GATED[chan]["IP"], cmap=cmocean.cm.phase, vmin=-vmax2, vmax=vmax2)
                elif phase == 2:
                    im1 = ax1.pcolormesh(self.GATED[chan]["GTT"], self.GATED[chan]["QQ"], self.GATED[chan]["CA"], cmap=dcmap, vmin=-vmax1, vmax=vmax1)
                    #XS = self.bootstrap_sigma(pulse, chan)
                    #im2 = ax2.pcolormesh(self.GATED[chan]["GTT"], self.GATED[chan]["QQ"], self.GATED[chan]["NR"], cmap=cmap, vmin=-vmax2, vmax=vmax2)
                    # bootstrap resample
#                     nt = len(self.GATED[chan]["GT"])
#                     nb = 5000
#                     XS = np.zeros( (nb, nt) )
#                     for ii in range(nb):
#                         #XS = []
#                         for it in range(nt):
#                             if self.GATED[chan]["isum"][it] < 8:
#                                 XS[ii, it] = self.sigma[pulse][chan]
#                             else:
#                                 if it == 0:                
#                                     X = self.bootstrap_resample( np.concatenate( (self.GATED[chan]["NR"][:,it], self.GATED[chan]["NR"][:,it+1], \
#                                         self.GATED[chan]["NR"][:,it+2], self.GATED[chan]["NR"][:,it+3] ) ), n=nt )
#                                 if it == 1:                
#                                     X = self.bootstrap_resample( np.concatenate( (self.GATED[chan]["NR"][:,it-1], self.GATED[chan]["NR"][:,it], \
#                                         self.GATED[chan]["NR"][:,it+1], self.GATED[chan]["NR"][:,it+2] ) ), n=nt )
#                                 elif it ==  nt-2:              
#                                     X = self.bootstrap_resample( np.concatenate( (self.GATED[chan]["NR"][:,it+1], self.GATED[chan]["NR"][:,it], \
#                                         self.GATED[chan]["NR"][:,it-1], self.GATED[chan]["NR"][:,it-2] ) ), n=nt )
#                                 elif it ==  nt-1:              
#                                     X = self.bootstrap_resample( np.concatenate( (self.GATED[chan]["NR"][:,it], self.GATED[chan]["NR"][:,it-1], \
#                                         self.GATED[chan]["NR"][:,it-2], self.GATED[chan]["NR"][:,it-3] ) ), n=nt )
#                                 else:              
#                                     X = self.bootstrap_resample( np.concatenate( (self.GATED[chan]["NR"][:,it-2] , self.GATED[chan]["NR"][:,it-1], \
#                                         self.GATED[chan]["NR"][:,it], self.GATED[chan]["NR"][:,it+1], self.GATED[chan]["NR"][:,it+2] )), n=nt )
#                                 XS[ii,it] = np.std(X)
                        #if ii == 0:
                        #    ax2.plot( self.GATED[chan]["GT"], XS[ii], '-', linewidth=1, markersize=4, alpha=.5, color='lightgrey', label = "bootstrap sim" )
                        #else:    
                        #    ax2.plot( self.GATED[chan]["GT"], XS[ii], '-', linewidth=1, markersize=4, alpha=.5, color='lightgrey'  )
                    ax2.plot( self.GATED[chan]["GT"], np.std(self.GATED[chan]["NR"], axis=0), color='darkgrey', linewidth=2, label="gate std" )
                    ax2.plot( self.GATED[chan]["GT"], np.average(self.GATED[chan]["BN"], axis=0), color='black', linewidth=2, label="boot average" )
                    #ax2.plot( np.tile(self.GATED[chan]["GT"], (5000,1) ), XS, '.', color='lightgrey', linewidth=1, alpha=.5 )
                    #ax2.plot( self.GATED[chan]["GT"], np.average(XS, axis=0), color='black', linewidth=2, label="bootstrap avg." )
                    #ax2.plot( self.GATED[chan]["GT"], np.median(XS, axis=0), color='black', linewidth=2, label="bootstrap med." )
                    ax2.legend()

                im1.set_edgecolor('face')
                if phase != 2:
                    im2.set_edgecolor('face')

                plt.setp(ax1.get_xticklabels(), visible=False)
        
                ax1.set_ylim( np.min(self.GATED[chan]["QQ"]), np.max(self.GATED[chan]["QQ"]) )
                
                if phase != 2:
                    ax2.set_ylim( np.min(self.GATED[chan]["QQ"]), np.max(self.GATED[chan]["QQ"]) )

                ax1.set_xlim( np.min(self.GATED[chan]["GTT"]), np.max(self.GATED[chan]["GTT"]) )
                ax2.set_xlim( np.min(self.GATED[chan]["GTT"]), np.max(self.GATED[chan]["GTT"]) )
        
                ax1.set_yscale('log')
                ax2.set_yscale('log')
                
                qlabs = np.append(np.concatenate( ( self.GATED[chan]["QQ"][0:1], self.GATED[chan]["QQ"][9::10] )), self.GATED[chan]["QQ"][-1] ) 
                ax1.yaxis.set_ticks( qlabs ) # np.append(np.concatenate( (QQ[0:1],QQ[9::10] )), QQ[-1] ) )
                if phase != 2:
                    ax2.yaxis.set_ticks( qlabs ) #np.append(np.concatenate( (QQ[0:1],QQ[9::10] )), QQ[-1] ) )
                #formatter = matplotlib.ticker.LogFormatter(10, labelOnlyBase=False)
                formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: str((round(x,1)))) 
                
                ax1.set_xscale('log')
                ax2.set_xscale('log')
                
                ax1.yaxis.set_major_formatter(formatter) #matplotlib.ticker.FormatStrFormatter('%d.1'))
                ax2.yaxis.set_major_formatter(formatter) #matplotlib.ticker.FormatStrFormatter('%d.1'))

                ax1.xaxis.set_major_formatter(formatter) #matplotlib.ticker.FormatStrFormatter('%d.1'))
                ax2.xaxis.set_major_formatter(formatter) #matplotlib.ticker.FormatStrFormatter('%d.1'))
                
                if ichan == 0:
                    ax1.set_ylabel(r"$q$ ( $\mathrm{A}\cdot\mathrm{s}$)", fontsize=fs)
                    if phase == 2:
                        ax2.set_ylabel(r"noise est. (nV)", fontsize=fs)
                    else:
                        ax2.set_ylabel(r"$q$ ( $\mathrm{A}\cdot\mathrm{s}$)", fontsize=fs)
                else:
                    plt.setp(ax1.get_yticklabels(), visible=False)
                    plt.setp(ax2.get_yticklabels(), visible=False)
        
                ax2.set_xlabel(r"$t-\tau_p$  (ms)", fontsize=fs)
                ichan += 1 

        #canvas.fig.subplots_adjust(hspace=.1, wspace=.05, left=.075, right=.925 )#left=None, bottom=None, right=None, top=None, wspace=None, hspace=None) 
        #canvas.fig.tight_layout()
        #canvas.draw()
        canvas.fig.subplots_adjust(hspace=.15, wspace=.05, left=.15, right=.9, bottom=.1, top=.9 )#left=None, bottom=None, right=None, top=None, wspace=None, hspace=None) 
        tick_locator = MaxNLocator(nbins=5)

        cb1 = canvas.fig.colorbar(im1, ax=axes[0::2], format='%1.0f', orientation='horizontal', shrink=.35, aspect=30)
        cb1.ax.tick_params(axis='both', which='major', labelsize=fs)
        cb1.set_label("$\mathcal{V}_N$ (nV)", fontsize=fs)
        #cb1.locator = tick_locator
        #cb1.update_ticks()

        if phase != 2:
            cb2 = canvas.fig.colorbar(im2, ax=axes[1::2], format='%1.0f', orientation='horizontal', shrink=.35, aspect=30, pad=.2)
            cb2.ax.tick_params(axis='both', which='major', labelsize=fs)
            cb2.set_label("$\mathcal{V}_N$ (nV)", fontsize=fs)

            cb2.locator = tick_locator
            cb2.update_ticks()

        canvas.draw()
        self.doneTrigger.emit() 
        

    def FDSmartStack(self, cv, canvas):
        from matplotlib.colors import LogNorm
        from matplotlib.ticker import MaxNLocator
        """
            Currently this stacks 4-phase second pulse data only, we need to generalise  
        """

        try:
            canvas.fig.clear()
        except:
            pass
        
    
        self.dataCubeFFT( )

        # canvas.ax1 = canvas.fig.add_axes([.1,  .1, .8, .8])
        canvas.ax1 = canvas.fig.add_axes([.1,  .1, .2, .8])
        canvas.ax2 = canvas.fig.add_axes([.325,  .1, .2, .8])
        canvas.ax3 = canvas.fig.add_axes([.55,  .1, .2, .8])
        canvas.ax4 = canvas.fig.add_axes([.815,  .1, .05, .8]) #cb 
        canvas.ax1.tick_params(axis='both', which='major', labelsize=8)
        canvas.ax2.tick_params(axis='both', which='major', labelsize=8)
        canvas.ax3.tick_params(axis='both', which='major', labelsize=8)
        canvas.ax4.tick_params(axis='both', which='major', labelsize=8)
        canvas.ax1.set_ylabel("pulse index", fontsize=8)
        canvas.ax1.set_xlabel(r"$\omega$ bin", fontsize=8)
        canvas.ax2.set_xlabel(r"$\omega$ bin", fontsize=8)
        canvas.ax3.set_xlabel(r"$\omega$ bin", fontsize=8)
        canvas.ax2.yaxis.set_ticklabels([])
        canvas.ax3.yaxis.set_ticklabels([])

        #canvas.ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  

#         # Look at pulses 
#         for pulse in self.DATADICT["PULSES"]:
#             for istack in self.DATADICT["stacks"]:
#                 for ipm in range(0,3):
#                     canvas.ax1.plot( self.DATADICT[pulse]["CURRENT"][ipm][istack] , label="istack "+str(istack) + " ipm=" + str(ipm) + pulse  )
#                     canvas.draw()

        # Create Container for stacks
        

        # sandbox determine pulse sequence again
        for pulse in self.DATADICT["PULSES"]:
            for ichan in self.DATADICT[pulse]["chan"]:
            #for ipm in range(10,11):
                CONTAINER = {}
                CONTAINER["Cycle 1"] = [] # These are actually subtracted cycles... v+ - v
                CONTAINER["Cycle 2"] = []
                for istack in self.DATADICT["stacks"]:
                    #canvas.ax1.clear()
                    ipm = 8
                    #for ipm in range(self.DATADICT["nPulseMoments"]):
                    #canvas.ax1.matshow( np.real(self.DATADICT[pulse][ichan]["FFT"][istack]), aspect='auto' )
                    #canvas.draw()
                    if not istack%4%4:
                        # phase cycle 4, aligned with 1 after sub
                        CONTAINER["Cycle 1"].append(-self.DATADICT[pulse][ichan]["FFT"][istack])
                        #canvas.ax1.plot(  self.DATADICT[pulse]["TIMES"], -self.DATADICT[pulse][ichan][ipm][istack], label="istack "+str(istack)+ " " + pulse )
                    elif not istack%4%3:
                        # phase cycle 3, aligned with 2 after sub
                        CONTAINER["Cycle 2"].append(-self.DATADICT[pulse][ichan]["FFT"][istack])
                        #canvas.ax1.plot(  self.DATADICT[pulse]["TIMES"], -self.DATADICT[pulse][ichan][ipm][istack], label="istack "+str(istack)+ " " + pulse )
                    elif not istack%4%2:
                        # phase cycle 2
                        CONTAINER["Cycle 2"].append( self.DATADICT[pulse][ichan]["FFT"][istack])
                        #canvas.ax1.plot(  self.DATADICT[pulse]["TIMES"], self.DATADICT[pulse][ichan][ipm][istack], label="istack "+str(istack)+ " " + pulse )
                    else:
                        # phase cycle 1
                        CONTAINER["Cycle 1"].append( self.DATADICT[pulse][ichan]["FFT"][istack])
                        #canvas.ax1.plot(  self.DATADICT[pulse]["TIMES"], self.DATADICT[pulse][ichan][ipm][istack], label="istack "+str(istack)+ " " + pulse )
                    #canvas.ax1.matshow(np.array(np.average(self.DATADICT[pulse][ichan]["FFT"]), axis=2), aspect='auto' )
                        #canvas.ax1.plot( self.DATADICT[pulse]["PULSE_TIMES"], self.DATADICT[pulse]["CURRENT"][ipm][istack] , color='black', label="istack "+str(istack)  )
                        #canvas.ax1.plot( self.DATADICT[pulse]["CURRENT"][ipm][istack] , label="istack "+str(istack) + " iFID" + str(iFID)  )
                    #canvas.ax1.plot(  self.DATADICT[pulse]["TIMES"], self.DATADICT[pulse][ichan][ipm][istack], label="istack "+str(istack)+ " " + pulse )

        #canvas.ax1.legend(prop={'size':6})
        #canvas.draw()

        # Boostrap 
        # stack. 
        #scipy.random.shuffle(x)
        


        # Stack and calculate the pooled variance (http://en.wikipedia.org/wiki/Pooled_variance)
        """ All this phase cycling wreaks havoc on a normal calculation of std. and variance. Instead, we resort to calculating 
            a pooled variance. In this assumption is that the precision of the measurment is constant. This is a poor choice for 
            any type of moving sensor. 
        """         
        # if a window filter has been applied        
        #self.WINDOW 
        #self.IWindowStart 
        #self.iWindowEnd 
        #self.FFTtimes
        CONTAINER = .5*(np.array(CONTAINER["Cycle 2"]) - np.array(CONTAINER["Cycle 1"]))
        print ("container shape", np.shape( CONTAINER), self.iWindowStart+1, self.iWindowEnd-1)
        dmin = np.min(np.abs(np.average(np.array(CONTAINER)[:,:,self.iWindowStart+1:self.iWindowEnd-1], axis=0)))
        dmax = np.max(np.abs(np.average(np.array(CONTAINER)[:,:,self.iWindowStart+1:self.iWindowEnd-1], axis=0)))
        mn = canvas.ax1.matshow( 20.*np.log10(np.abs(np.average(np.array(CONTAINER)[:,:, self.iWindowStart+1:self.iWindowEnd-1], axis=0))), aspect='auto', vmin=-120, vmax=-40)
        #mn = canvas.ax1.matshow(20.*np.log10(XA[:,istart:iend+1]), aspect='auto', vmax=-40, vmin=-120) #, norm=LogNorm())
        canvas.ax2.matshow( 20*np.log10(np.std(np.real(np.array(CONTAINER)[:,:,self.iWindowStart+1:self.iWindowEnd-1]), axis=0)), aspect='auto', vmin=-120, vmax=-40)
        canvas.ax3.matshow( 20*np.log10(np.std(np.imag(np.array(CONTAINER)[:,:,self.iWindowStart+1:self.iWindowEnd-1]), axis=0)), aspect='auto', vmin=-120, vmax=-40)
        #canvas.ax1.legend(prop={'size':6})

        cb1 = mpl.colorbar.Colorbar(canvas.ax4, mn)
        cb1.ax.tick_params(labelsize=8) 
        cb1.set_label("power [dB]", fontsize=8) 
        canvas.ax1.xaxis.set_major_locator(MaxNLocator(4))
        canvas.ax2.xaxis.set_major_locator(MaxNLocator(4))
        canvas.ax3.xaxis.set_major_locator(MaxNLocator(4))
        canvas.draw()

        self.doneTrigger.emit() 

    def effectivePulseMoment(self, cv, canvas):
        
        canvas.reAxH(2)
        nstack = len(self.DATADICT["stacks"]) 
        #canvas.ax1.set_yscale('log')

        for pulse in self.DATADICT["PULSES"]:
            self.DATADICT[pulse]["qeff"] = {}
            self.DATADICT[pulse]["q_nu"] = {}
            for ipm in range(self.DATADICT["nPulseMoments"]):
                self.DATADICT[pulse]["qeff"][ipm] = {}
                self.DATADICT[pulse]["q_nu"][ipm] = {}
                #canvas.ax1.clear()
                #scolours = np.array( (   np.linspace(0.8,0.4,len(self.DATADICT["stacks"])), \
                #                         np.linspace(0.0,0.6,len(self.DATADICT["stacks"])), \
                #                         np.linspace(0.6,0.0,len(self.DATADICT["stacks"])) )   
                #                   ).T

                #scolours = plt.cm.Spectral(np.linspace(0,1,len(self.DATADICT["stacks"])))
                #scolours = plt.cm.Blues(np.linspace(0,1,1.5*len(self.DATADICT["stacks"])))
                scolours = cmocean.cm.ice(np.linspace(0,1,int(1.5*len(self.DATADICT["stacks"]))))
                iistack = 0
                for istack in self.DATADICT["stacks"]:
                    #self.DATADICT[pulse]["PULSE_TIMES"]
                    x = self.DATADICT[pulse]["CURRENT"][ipm][istack]
                    X = np.fft.rfft(x)
                    v = np.fft.fftfreq(len(x), self.dt)
                    v = v[0:len(X)]
                    v[-1] = np.abs(v[-1])

                    # calculate effective current/moment
                    I0 = np.abs(X)/len(X) 
                    qeff = I0 * (self.DATADICT[pulse]["PULSE_TIMES"][-1]-self.DATADICT[pulse]["PULSE_TIMES"][0])

                    # frequency plot
                    #canvas.ax1.set_title(r"pulse moment index " +str(ipm), fontsize=10)
                    #canvas.ax1.set_xlabel(r"$\nu$ [Hz]", fontsize=8)
                    #canvas.ax1.set_ylabel(r"$q_{eff}$ [A$\cdot$sec]", fontsize=8)
                    #canvas.ax1.plot(v, qeff, color=scolours[iistack] ) # eff current
                    
                    # time plot
                    canvas.ax1.plot(1e2*(self.DATADICT[pulse]["PULSE_TIMES"]-self.DATADICT[pulse]["PULSE_TIMES"][0]), x, color=scolours[iistack])

                    self.DATADICT[pulse]["qeff"][ipm][istack] = qeff
                    self.DATADICT[pulse]["q_nu"][ipm][istack] = v
                    iistack += 1
                #canvas.draw()
                        
                percent = int(1e2* (float)((istack)+ipm*self.DATADICT["nPulseMoments"]) / 
                                   (float)(len(self.DATADICT["PULSES"])*self.DATADICT["nPulseMoments"]*nstack))
                self.progressTrigger.emit(percent)

            canvas.ax1.set_xlabel("time (ms)", fontsize=10)
            canvas.ax1.set_ylabel("current (A)", fontsize=10)
            #canvas.draw()

        self.plotQeffNu(cv, canvas.ax2)

        deSpine(canvas.ax1)        
        deSpine(canvas.ax2)        

        canvas.fig.tight_layout()
        canvas.draw()
        self.doneTrigger.emit() 

    def plotQeffNu(self, cv, ax):

        ####################################
        # TODO label fid1 and fid2, and make a legend, and colour by pulse 
        nstack = len(self.DATADICT["stacks"])
        iFID = 0 
        for pulse in self.DATADICT["PULSES"]:
            self.DATADICT[pulse]["Q"] = np.zeros( (self.DATADICT["nPulseMoments"], len(self.DATADICT["stacks"])) )
            ilabel = True
            for ipm in range(self.DATADICT["nPulseMoments"]):
                #scolours = np.array([0.,0.,1.])
                scolours = cmocean.cm.ice(np.linspace(0,1,int(1.5*len(self.DATADICT["stacks"]))))
                #scolours = plt.cm.Spectral(np.linspace(0,1,len(self.DATADICT["stacks"])))
                #scolours = plt.cm.Spectral(np.linspace(0,1,len(self.DATADICT["stacks"])))
                istack = 0
                for stack in self.DATADICT["stacks"]:
                    # find index 
                    icv = int(round(cv / self.DATADICT[pulse]["q_nu"][ipm][stack][1]))
                    self.DATADICT[pulse]["Q"][ipm,istack] = self.DATADICT[pulse]["qeff"][ipm][stack][icv]
                    if ilabel:
                        ax.scatter(ipm, self.DATADICT[pulse]["qeff"][ipm][stack][icv], facecolors='none', edgecolors=scolours[istack], label=(str(pulse)))
                        ilabel = False
                    else:    
                        ax.scatter(ipm, self.DATADICT[pulse]["qeff"][ipm][stack][icv], facecolors='none', edgecolors=scolours[istack])
                    #scolours += np.array((0,1./(nstack+1),-1/(nstack+1.)))

                    percent = int(1e2* (float)((istack)+ipm*self.DATADICT["nPulseMoments"]) / 
                                       (float)(len(self.DATADICT["PULSES"])*self.DATADICT["nPulseMoments"]*nstack))
                    self.progressTrigger.emit(percent)
                    istack += 1
            iFID += 1
        ax.set_xlabel(r"pulse moment index", fontsize=10)
        ax.set_ylabel(r"$q_{eff}$ (A$\cdot$sec)", fontsize=10)
        ax.set_yscale('log')
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.legend(loc='upper right', scatterpoints = 1, prop={'size':10})

    def enableDSP(self):
        self.enableDSPTrigger.emit() 

    def adaptiveFilter(self, M, flambda, truncate, mu, PCA, canvas):

        canvas.reAx2(shx=False, shy=False)
        # ax1 is top plot of filter taps 
        # ax2 is bottom plot of conditioned signal 
        
        if truncate:
            itrunc =(int) ( round( 1e-3*truncate*self.samp ) )

        print( "adaptive filter size", 1e3*self.dt*M, " [ms]" )       
 
        Filt = adapt.AdaptiveFilter(flambda)
        H = {} 
        for pulse in self.DATADICT["PULSES"]:
            H[pulse] = {}
            for ichan in self.DATADICT[pulse]["chan"]:
                H[pulse][ichan] = np.zeros(M*len( self.DATADICT[pulse]["rchan"] ))
        
        iFID = 0 
        # original ordering... 
        #for pulse in self.DATADICT["PULSES"]:
        #    for ipm in range(self.DATADICT["nPulseMoments"]):
        #        for istack in self.DATADICT["stacks"]:
        # This order makes more sense, same as data collection, verify
        for istack in self.DATADICT["stacks"]:
            for ipm in range(self.DATADICT["nPulseMoments"]):
                for pulse in self.DATADICT["PULSES"]:
                    canvas.softClear()
                    mmax = 0
                    for ichan in self.DATADICT[pulse]["chan"]:
                        canvas.ax2.plot( self.DATADICT[pulse]["TIMES"], 1e9* self.DATADICT[pulse][ichan][ipm][istack], alpha=.5) 
                        mmax = max(mmax, np.max(1e9*self.DATADICT[pulse][ichan][ipm][istack])) 
                    canvas.ax2.set_ylim(-mmax, mmax) 
                    canvas.ax2.set_prop_cycle(None)
                    for ichan in self.DATADICT[pulse]["chan"]:
                        #H = np.zeros(M)
                        RX = []
                        for irchan in self.DATADICT[pulse]["rchan"]:
                            RX.append(self.DATADICT[pulse][irchan][ipm][istack][::-1])
                        # Reset each time? 
                        #H[pulse][ichan] *= 0
                        #if all(H[pulse][ichan]) == 0:
                        if False: 
                            ####################################################################################
                            # Padasip adaptive filter implimentations, do not allow for variable filter length
                            ####################################################################################
                            # identification                                                                   #
                            #f = pa.filters.FilterRLS(n=len(self.DATADICT[pulse]["rchan"]), mu=0.99, w="zeros") #
                            #f = pa.filters.FilterGNGD(n=len(self.DATADICT[pulse]["rchan"]), mu=0.1)           #                          # Nope
                            #f = pa.filters.FilterLMS(n=len(self.DATADICT[pulse]["rchan"]), mu=0.1)            #                          # NOPE
                            #f = pa.filters.AdaptiveFilter(model="NLMS", n=len(self.DATADICT[pulse]["rchan"]), mu=0.1, w="random")        # NOPE  
                            #f = pa.filters.AdaptiveFilter(model="GNGD", n=len(self.DATADICT[pulse]["rchan"]), mu=0.1)                    # horrendous
                            #f = pa.filters.FilterNLMF(n=len(self.DATADICT[pulse]["rchan"]), mu=0.005, w="random")                        # BAD
                            
                            #f = pa.filters.FilterSSLMS(n=len(self.DATADICT[pulse]["rchan"]), mu=0.01, w="zeros")                         # pretty good
                            f = pa.filters.FilterNSSLMS(n=len(self.DATADICT[pulse]["rchan"]), mu=0.1, w="zeros")                          # pretty good 

                            y, e, H[pulse][ichan] = f.run(self.DATADICT[pulse][ichan][ipm][istack][::-1], np.array(RX).T)    #
                            ####################################################################################
                            e = self.DATADICT[pulse][ichan][ipm][istack][::-1] - y
                        elif True:
                            # check for change in filter coefficients and rerun if things are changing too rapidly, 
                            #       this is especially true for the first run 
                            hm1 = np.copy(H[pulse][ichan]) 
                            [e, H[pulse][ichan]] = Filt.adapt_filt_Ref( self.DATADICT[pulse][ichan][ipm][istack][::-1],\
                                                     RX,\
                                                     M, mu, PCA, flambda, H[pulse][ichan])
                            iloop = 0
                            #while False: 
                            while (np.linalg.norm( H[pulse][ichan] - hm1) > .05): # threshold for recall 
                                hm1 = np.copy(H[pulse][ichan]) 
                                [e, H[pulse][ichan]] = Filt.adapt_filt_Ref( self.DATADICT[pulse][ichan][ipm][istack][::-1],\
                                                        RX,\
                                                        M, mu, PCA, flambda, H[pulse][ichan])
                                iloop += 1
                            print("Recalled ", iloop, "times with norm=", np.linalg.norm(hm1-H[pulse][ichan]))
                        else:
                            [e,H[pulse][ichan]] = Filt.adapt_filt_Ref( self.DATADICT[pulse][ichan][ipm][istack][::-1],\
                                                        RX,\
                                                        M, mu, PCA, flambda, H[pulse][ichan])
                        # replace
                        if truncate:
                            canvas.ax2.plot( self.DATADICT[pulse]["TIMES"][0:itrunc], 1e9* e[::-1][0:itrunc],\
                                label = pulse + " ipm=" + str(ipm) + " istack=" + str(istack) + " ichan="  + str(ichan))
                            self.DATADICT[pulse][ichan][ipm][istack] = e[::-1][0:itrunc]
                        else:
                            canvas.ax2.plot( self.DATADICT[pulse]["TIMES"], 1e9* e[::-1],\
                                label = pulse + " ipm=" + str(ipm) + " istack=" + str(istack) + " ichan="  + str(ichan))
                            self.DATADICT[pulse][ichan][ipm][istack] = e[::-1]
                           
     
                        #canvas.ax1.plot( H[pulse][ichan].reshape(-1, len(RX)) ) # , label="taps") 
                        canvas.ax1.plot( H[pulse][ichan][::-1].reshape(M, len(RX), order='F' ) ) #.reshape(-1, len(RX)) ) # , label="taps") 

                        canvas.ax2.legend(prop={'size':10}, loc='upper right')
                        #canvas.ax2.legend(prop={'size':6}, loc='upper right')

                        mh = np.max(np.abs( H[pulse][ichan] ))
                        canvas.ax1.set_ylim( -mh, mh )

                        canvas.ax2.set_xlabel(r"time (s)", fontsize=10)
                        canvas.ax2.set_ylabel(r"signal (nV)", fontsize=10)

                        canvas.ax1.set_xlabel(r"filter tap index", fontsize=10)
                        canvas.ax1.set_ylabel(r"tap amplitude", fontsize=10)

                    canvas.fig.tight_layout()
                    deSpine(canvas.ax1)
                    deSpine(canvas.ax2)

                    canvas.draw()

                    # truncate the reference channels too, in case you still need them for something. 
                    # Otherwise they are no longer aligned with the data 
                    for rchan in self.DATADICT[pulse]["rchan"]:
                        if truncate:
                            self.DATADICT[pulse][rchan][ipm][istack] = self.DATADICT[pulse][rchan][ipm][istack][0:itrunc]
                
                #percent = (int)(1e2*((float)(iFID*self.DATADICT["nPulseMoments"]+(ipm))/( len(self.DATADICT["PULSES"])*self.nPulseMoments)))
                percent = (int)(1e2*((float)(istack*self.DATADICT["nPulseMoments"]+(ipm))/( len(self.DATADICT["PULSES"])*self.nPulseMoments*(len(self.DATADICT["stacks"])+1) )))
                self.progressTrigger.emit(percent)
                            
#         # why is this loop here, istack is not part of rest?
#         for istack in self.DATADICT["stacks"]:
#             if truncate:
#                 self.DATADICT[pulse]["TIMES"] = self.DATADICT[pulse]["TIMES"][0:itrunc]
#                 percent = (int)(1e2*((float)(iFID*self.DATADICT["nPulseMoments"]+(ipm))/( len(self.DATADICT["PULSES"])*self.nPulseMoments)))
#                 self.progressTrigger.emit(percent)
#             iFID += 1 
        
        if truncate:
            self.DATADICT[pulse]["TIMES"] = self.DATADICT[pulse]["TIMES"][0:itrunc]
        
        self.doneTrigger.emit() 
        self.updateProcTrigger.emit()  

        #self.plotFT(canvas)

    def plotFT(self, canvas, istart=0, iend=0):

        try:
            canvas.fig.clear()
        except:
            pass

        canvas.ax1  = canvas.fig.add_axes([.1, .1, .65, .8])
        canvas.ax1c = canvas.fig.add_axes([.8, .1, .05, .8])
        canvas.ax1.tick_params(axis='both', which='major', labelsize=8)
        
        for pulse in self.DATADICT["PULSES"]:
            for istack in self.DATADICT["stacks"]:
                for ichan in self.DATADICT[pulse]["chan"]:
                    # FFT of stack
                    XA = np.zeros((self.DATADICT["nPulseMoments"] , len(self.DATADICT[pulse][ichan][0][istack])/2+1))
                    nu = np.fft.fftfreq(self.DATADICT[pulse][ichan][0][istack].size, d=self.dt)
                    nu[-1] *= -1
                    df = nu[1]
                    of = 0
                    if istart:
                        of = nu[istart]
                    def freqlabel(x, pos):
                        return  '%1.0f' %(of + x*df)
                    formatter = FuncFormatter(freqlabel)
                    canvas.ax1.clear()
                    for ipm in range(self.DATADICT["nPulseMoments"]):
                        X = np.fft.rfft(self.DATADICT[pulse][ichan][ipm][istack])
                        XA[ipm,:] = np.abs(X)   
                    if istart:
                        mn = canvas.ax1.matshow(20.*np.log10(XA[:,istart:iend+1]), aspect='auto', vmax=-40, vmin=-120) #, norm=LogNorm())
                    else:
                        mn = canvas.ax1.matshow(20.*np.log10(XA), aspect='auto', vmax=-40, vmin=-120) #, norm=LogNorm())
                    smin = np.min(20.*np.log10(XA))
                    smax = np.max(20.*np.log10(XA))
                    canvas.ax1.xaxis.set_major_formatter(formatter)
                    cb1 = mpl.colorbar.Colorbar(canvas.ax1c, mn)
                    cb1.ax.tick_params(labelsize=8) 
                    cb1.set_label("signal [dB]", fontsize=8) 
                    canvas.ax1.set_xlabel(r"$\nu$ [Hz]", fontsize=10)
                    canvas.ax1.set_ylabel(r"$q_{index}$", fontsize=10)
                    canvas.draw()

    def plotFT(self, canvas, istart=0, iend=0):

        try:
            canvas.fig.clear()
        except:
            pass

        canvas.ax1  = canvas.fig.add_axes([.1, .1, .65, .8])
        canvas.ax1c = canvas.fig.add_axes([.8, .1, .05, .8])
        canvas.ax1.tick_params(axis='both', which='major', labelsize=8)
        
        for pulse in self.DATADICT["PULSES"]:
            for istack in self.DATADICT["stacks"]:
                for ichan in self.DATADICT[pulse]["chan"]:
                    # FFT of stack
                    XA = np.zeros((self.DATADICT["nPulseMoments"] , len(self.DATADICT[pulse][ichan][0][istack])//2+1))
                    nu = np.fft.fftfreq(self.DATADICT[pulse][ichan][0][istack].size, d=self.dt)
                    nu[-1] *= -1
                    df = nu[1]
                    of = 0
                    if istart:
                        of = nu[istart]
                    def freqlabel(x, pos):
                        return  '%1.0f' %(of + x*df)
                    formatter = FuncFormatter(freqlabel)
                    canvas.ax1.clear()
                    for ipm in range(self.DATADICT["nPulseMoments"]):
                        X = np.fft.rfft(self.DATADICT[pulse][ichan][ipm][istack])
                        XA[ipm,:] = np.abs(X)   
                    if istart:
                        mn = canvas.ax1.matshow(20.*np.log10(XA[:,istart:iend+1]), aspect='auto', vmax=-40, vmin=-120, cmap='viridis') #, norm=LogNorm())
                    else:
                        mn = canvas.ax1.matshow(20.*np.log10(XA), aspect='auto', vmax=-40, vmin=-120, cmap='viridis') #, norm=LogNorm())
                    canvas.ax1.xaxis.set_major_formatter(formatter)
                    cb1 = mpl.colorbar.Colorbar(canvas.ax1c, mn)
                    cb1.ax.tick_params(labelsize=8) 
                    cb1.set_label("signal [dB]", fontsize=8) 
                    canvas.ax1.set_xlabel(r"$\nu$ [Hz]", fontsize=10)
                    canvas.ax1.set_ylabel(r"$q_{index}$", fontsize=10)
                    canvas.draw()


    def dataCubeFFT(self):
        """
            Performs FFT on entire cube of DATA, and REFERENCE channels, but not pulse currents, 
            Results are saved to a new field in the data structure

            The GMR varies phase as a function of pulse moment index, so that the first pusle moment is zero phase, 
            the second is pi/2 the third is zero. This method corrects for this, so that all pulse moments are in phase. 
        
            Technically we may not want to do this, if there is some system response that this cycles away, and we lose track of 
            how many of each cycle we have, could this be problomatic? I think it will come out in the wash as we keep track of the 
            rest of the phase cycles. Holy phase cycling batman. 
        """        
        for pulse in self.DATADICT["PULSES"]:
            for ichan in np.append(self.DATADICT[pulse]["chan"], self.DATADICT[pulse]["rchan"]):
                # FFT of stack
                self.DATADICT[pulse][ichan]["FFT"] = {}
                self.DATADICT[pulse][ichan]["FFT"]["nu"] = np.fft.fftfreq(self.DATADICT[pulse][ichan][0][self.DATADICT["stacks"][0]].size, d=self.dt)
                self.DATADICT[pulse][ichan]["FFT"]["nu"][-1] *= -1
                for istack in self.DATADICT["stacks"]:
                    self.DATADICT[pulse][ichan]["FFT"][istack] = np.zeros((self.DATADICT["nPulseMoments"] , len(self.DATADICT[pulse][ichan][0][istack])//2+1), dtype=complex)
                    for ipm in range(self.DATADICT["nPulseMoments"]):
                        # Mod works for FID pulse sequences, TODO generalize this for 4 phase T1, etc..
                        mod = (-1)**(ipm%2) * (-1)**(istack%2)
                        self.DATADICT[pulse][ichan]["FFT"][istack][ipm,:] = np.fft.rfft( self.DATADICT[pulse][ichan][ipm][istack] )
                        #if ipm%2:
                            # odd, phase cycled from previous
                        #    self.DATADICT[pulse][ichan]["FFT"][istack][ipm,:] = np.fft.rfft(-self.DATADICT[pulse][ichan][ipm][istack])
                        #else:
                            # even, we define as zero phase, first pulse moment has this
                        #    self.DATADICT[pulse][ichan]["FFT"][istack][ipm,:] = np.fft.rfft(self.DATADICT[pulse][ichan][ipm][istack])

    def adaptiveFilterFD(self, ftype, band, centre, canvas):
        
        try:
            canvas.fig.clear()
        except:
            pass

        canvas.ax1  = canvas.fig.add_axes([.1, .5, .7, .4])
        canvas.ax1c = canvas.fig.add_axes([.85, .5, .05, .4])
        canvas.ax1.tick_params(axis='both', which='major', labelsize=8)
        #canvas.ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
        
        canvas.ax2 = canvas.fig.add_axes([.1, .05, .7, .4])
        canvas.ax2c = canvas.fig.add_axes([.85, .05, .05, .4])
        canvas.ax2.tick_params(axis='both', which='major', labelsize=8)
        #canvas.ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  

        self.dataCubeFFT()

        Filt = adapt.AdaptiveFilter(0.)
        for pulse in self.DATADICT["PULSES"]:
            # Compute window function and dimensions      
            [WINDOW, nd, wstart, wend, dead, idead] = self.computeWindow(pulse, band, centre, ftype) 
            for istack in self.DATADICT["stacks"]:
                for ichan in self.DATADICT[pulse]["chan"]:
                    # FFT of stack
                    nd = len(self.DATADICT[pulse][ichan][0][istack])
                    XX = np.zeros((self.DATADICT["nPulseMoments"] , len(self.DATADICT[pulse][ichan][0][istack])//2+1), dtype=complex)
                    nu = np.fft.fftfreq(self.DATADICT[pulse][ichan][0][istack].size, d=self.dt)
                    nu[-1] *= -1
                    #nu = self.DATADICT[pulse][ichan]["FFT"]["nu"]
                    def freqlabel(x, pos):
                        return  '%1.0f' %((wstart)*nu[1] + x*nu[1])
                    formatter = FuncFormatter(freqlabel)
                    canvas.ax1.clear()
                    for ipm in range(self.DATADICT["nPulseMoments"]):
                        X = np.fft.rfft(self.DATADICT[pulse][ichan][ipm][istack])
                        XX[ipm,:] = X   

                    XX = XX*WINDOW
                    XX = XX[:,wstart:wend]
                    smin = np.min(20.*np.log10(np.abs(XX)))
                    smax = np.max(20.*np.log10(np.abs(XX)))
                    #if smin != smin:
                    smax = -40
                    smin = -120
                    mn = canvas.ax1.matshow(20.*np.log10(np.abs(XX)), aspect='auto', vmin=smin, vmax=smax) #, norm=LogNorm())

                    canvas.ax1.xaxis.set_major_formatter(formatter)
                    cb1 = mpl.colorbar.Colorbar(canvas.ax1c, mn) 
                
                    RX = []
                    for ichan in self.DATADICT[pulse]["rchan"]:
                        R = np.zeros((self.DATADICT["nPulseMoments"] , len(self.DATADICT[pulse][ichan][0][istack])//2+1), dtype=complex)
                        for ipm in range(self.DATADICT["nPulseMoments"]):
                            R[ipm,:] = np.fft.rfft(self.DATADICT[pulse][ichan][ipm][istack]) 
                        RX.append(R[:,wstart:wend])
                    XC = Filt.transferFunctionFFT(XX, RX)

                    # TODO inverse FFT, but we need to map back to origional matrix size
                    #for ichan in self.DATADICT[pulse]["chan"]:
                    #    for ipm in range(self.DATADICT["nPulseMoments"]):
                    #        self.DATADICT[pulse][ichan][ipm][istack] = np.fft.irfft(XC[] , nd)
                    mc = canvas.ax2.matshow(20.*np.log10(np.abs(XC)), aspect='auto', vmin=smin, vmax=smax) #, norm=LogNorm())
                    cb2 = mpl.colorbar.Colorbar(canvas.ax2c, mc) 
                    cmin = np.min(20.*np.log10(np.abs(XC)))
                    cmax = np.max(20.*np.log10(np.abs(XC)))
                    canvas.ax2.xaxis.set_major_formatter(formatter)
                    #canvas.ax2.colorbar(mn)
                    canvas.draw()

        ##############################3
        # TODO inverse FFT to get the damn data back!!!
#                 self.progressTrigger.emit(percent) 
#                             #label = "iFID="+str(iFID) + " ipm=" + str(ipm) + " istack=" + str(istack) + " ichan="  + str(ichan))
        self.doneTrigger.emit() 
        

    def findSpikes(self, x, width, threshold, rollOn):
        import scipy.ndimage as im
        spikes = np.zeros( len(x) )
        med = im.median_filter(x, width,mode='nearest')
        std = np.std(x)
        spikes = (np.abs(x-med) > threshold * std)
        return np.array(np.where(spikes[rollOn::])) + rollOn

#     def despike(self, width, threshold, itype, rollOn, win, canvas):
#         from scipy import interpolate
#         """ This was a stab at a despike filter. Better results were achieved using the SmartStack approach 
#         """    
#         try:
#             canvas.fig.clear()
#         except:
#             pass
#         
#         canvas.ax1 = canvas.fig.add_axes([.125,.1,.725,.8])
#         canvas.ax1.tick_params(axis='both', which='major', labelsize=8)
#         canvas.ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
#         iFID = 0 
#         for pulse in self.DATADICT["PULSES"]:
#             for ipm in range(self.DATADICT["nPulseMoments"]):
#                 for istack in self.DATADICT["stacks"]:
#                     canvas.ax1.clear()
#                     for ichan in np.append(self.DATADICT[pulse]["chan"], self.DATADICT[pulse]["rchan"]):
#                         x = self.findSpikes(self.DATADICT[pulse][ichan][ipm][istack], width, threshold, rollOn)             
#                         canvas.ax1.plot( self.DATADICT[pulse]["TIMES"],  self.DATADICT[pulse][ichan][ipm][istack], 
#                             label = pulse + " ipm=" + str(ipm) + " istack=" + str(istack) + " ichan="  + str(ichan))
#                         canvas.ax1.plot( self.DATADICT[pulse]["TIMES"][x], self.DATADICT[pulse][ichan][ipm][istack][x], '.', color='red' , markersize=6 )
# 
#                         FIXED = np.zeros(len(x[0]))
#                         ii = 0
#                         for spike in np.array(x[0]).tolist():
#                             f = interpolate.interp1d(np.delete(self.DATADICT[pulse]["TIMES"][spike-win/2:spike+win/2], x[0]-(spike-win/2)), \
#                                                      np.delete(self.DATADICT[pulse][ichan][ipm][istack][spike-win/2:spike+win/2], x[0]-(spike-win/2)), itype)
#                             FIXED[ii] = f(self.DATADICT[pulse]["TIMES"][spike])
#                             ii += 1
#                         canvas.ax1.plot( self.DATADICT[pulse]["TIMES"][x[0]] , FIXED, '.', color='black' , markersize=4 )
#                         self.DATADICT[pulse][ichan][ipm][istack][x[0]] =  FIXED 
# 
#                     canvas.ax1.legend(prop={'size':6})
#                     canvas.draw() 
#                 percent = (int)(1e2*((float)(iFID*self.DATADICT["nPulseMoments"]+(ipm))/( len(self.DATADICT["PULSES"])*self.nPulseMoments)))
#                 self.progressTrigger.emit(percent) 
#             iFID += 1
#         self.doneTrigger.emit() 

    def designFilter(self, cf, PB, SB, gpass, gstop, ftype, canvas):
        ''' cf is central frequency
            pb is pass band
            sb is stop band
        '''
        TS = (cf) / (.5/self.dt) 
        PB = PB / (.5/self.dt)   # 1/2 width pass band Muddy Creek
        SB = SB / (.5/self.dt)   # 1/2 width stop band Muddy Creek
        # if butterworth
        #[bord, wn] = signal.buttord([TS-PB,TS+PB], [TS-SB,TS+SB], 1e-1, 5.)
        if ftype=="Butterworth":
            [bord, wn] = signal.buttord([TS-PB,TS+PB], [TS-SB,TS+SB], gpass, gstop)
            [self.filt_b, self.filt_a] = signal.butter(bord, wn, btype='bandpass', output='ba')
            [self.filt_z, self.filt_p, self.filt_k] = signal.butter(bord, wn, btype='band', output='zpk')
        elif ftype == "Chebychev Type II":
            [bord, wn] = signal.cheb2ord([TS-PB,TS+PB], [TS-SB,TS+SB], gpass, gstop)
            [self.filt_b, self.filt_a] = signal.cheby2(bord, gstop, wn, btype='bandpass', output='ba')
            [self.filt_z, self.filt_p, self.filt_k] = signal.cheby2(bord, gstop, wn, btype='band', output='zpk')
        elif ftype == "Elliptic":
            [bord, wn] = signal.ellipord([TS-PB,TS+PB], [TS-SB,TS+SB], gpass, gstop)
            [self.filt_b, self.filt_a] = signal.ellip(bord, gpass, gstop, wn, btype='bandpass', output='ba')
            [self.filt_z, self.filt_p, self.filt_k] = signal.ellip(bord, gpass, gstop, wn, btype='band', output='zpk')
        
        # if cheby2
        impulse = self.mfreqz2(self.filt_b, self.filt_a, canvas)  
        self.fe = -5
        for it in range(len(impulse[0])):
            if abs(impulse[1][0][it][0]) >= .1 * gpass:# gpass:
                self.fe = impulse[0][it]
         
        canvas.draw() 
        return [bord, self.fe] 

    def downsample(self, truncate, dec, plot=False, canvas=None):
        """ Downsamples and truncates the raw signal.
            Args 
                truncate (float) : the length of the signal to truncate to  
                dec (int) : the decimation factor, 1 results in no downsampling
                plot (bool) : perform plots
                canvas : MPL axis for plotting  
        """
        
        if plot:
            fs = 10
            canvas.reAx2()
            canvas.ax1.set_ylabel(r"signal (nV)", fontsize=fs)
            canvas.ax2.set_xlabel(r"time (s)", fontsize=fs)
            canvas.ax2.set_ylabel(r"signal (nV)", fontsize=fs)

        self.samp /= dec
        self.dt   = 1./self.samp       

        iFID = 0  
        for pulse in self.DATADICT["PULSES"]:
            RSTIMES = self.DATADICT[pulse]["TIMES"][::dec]
            if truncate:
                itrunc = (int)( 1e-3*truncate*self.samp )
                RSTIMES = RSTIMES[0:itrunc]
            for ipm in range(self.DATADICT["nPulseMoments"]):
                for istack in self.DATADICT["stacks"]:
                    if plot:
                        canvas.softClear()
                    for ichan in np.append(self.DATADICT[pulse]["chan"], self.DATADICT[pulse]["rchan"]):
                        # trim off indices that don't divide evenly
                        ndi = np.shape(self.DATADICT[pulse][ichan][ipm][istack])[0]%dec
                        if ndi:
                            #[self.DATADICT[pulse][ichan][ipm][istack], RSTIMES] = signal.resample(self.DATADICT[pulse][ichan][ipm][istack][0:-ndi],\
                            #             len(self.DATADICT[pulse][ichan][ipm][istack][0:-ndi])//dec,\
                            #             self.DATADICT[pulse]["TIMES"][0:-ndi], window='hamm')
                            self.DATADICT[pulse][ichan][ipm][istack] = signal.decimate(self.DATADICT[pulse][ichan][ipm][istack], dec, n=None, ftype='iir', zero_phase=True)
                        else:
                            #[self.DATADICT[pulse][ichan][ipm][istack], RSTIMES] = signal.resample(self.DATADICT[pulse][ichan][ipm][istack],\
                            #             len(self.DATADICT[pulse][ichan][ipm][istack])//dec,\
                            #             self.DATADICT[pulse]["TIMES"], window='hamm')
                            self.DATADICT[pulse][ichan][ipm][istack] = signal.decimate(self.DATADICT[pulse][ichan][ipm][istack], dec, n=None, ftype='iir', zero_phase=True)
                        if truncate:
                            self.DATADICT[pulse][ichan][ipm][istack] = self.DATADICT[pulse][ichan][ipm][istack][0:itrunc]

                    if plot:
                        for ichan in self.DATADICT[pulse]["chan"]:
                            canvas.ax2.plot( RSTIMES, 1e9*self.DATADICT[pulse][ichan][ipm][istack], \
                                label = pulse + " ipm=" + str(ipm) + " istack=" + str(istack) + " ichan="  + str(ichan))
                        for ichan in self.DATADICT[pulse]["rchan"]:
                            canvas.ax1.plot( RSTIMES, 1e9*self.DATADICT[pulse][ichan][ipm][istack], \
                                label = pulse + " ipm=" + str(ipm) + " istack=" + str(istack) + " ichan="  + str(ichan))
                        canvas.ax1.legend(prop={'size':fs}, loc='upper right')
                        canvas.ax2.legend(prop={'size':fs}, loc='upper right')
                   
                        deSpine( canvas.ax1 ) 
                        deSpine( canvas.ax2 ) 
                        plt.setp(canvas.ax1.get_xticklabels(), visible=False)
                        canvas.fig.tight_layout()
                        canvas.draw() 

                percent = (int)(1e2*((float)(iFID*self.DATADICT["nPulseMoments"]+(ipm))/( len(self.DATADICT["PULSES"])*self.nPulseMoments)))
                self.progressTrigger.emit(percent)
            iFID += 1  
            self.DATADICT[pulse]["TIMES"] = RSTIMES

        #####################################
        # resample pulse data
        for pulse in self.DATADICT["PULSES"]:
            for ipm in range(self.DATADICT["nPulseMoments"]):
                for istack in self.DATADICT["stacks"]:
                    ndi = np.shape(self.DATADICT[pulse]["CURRENT"][ipm][istack])[0]%dec
                    if ndi:
                        [self.DATADICT[pulse]["CURRENT"][ipm][istack], RSPTIMES] = signal.resample(self.DATADICT[pulse]["CURRENT"][ipm][istack][0:-ndi],\
                                     len(self.DATADICT[pulse]["CURRENT"][ipm][istack][0:-ndi])//dec,\
                                     self.DATADICT[pulse]["PULSE_TIMES"][0:-ndi], window='hamm')
                    else:
                        [self.DATADICT[pulse]["CURRENT"][ipm][istack], RSPTIMES] = signal.resample(self.DATADICT[pulse]["CURRENT"][ipm][istack],\
                                     len(self.DATADICT[pulse]["CURRENT"][ipm][istack])//dec,\
                                     self.DATADICT[pulse]["PULSE_TIMES"], window='hamm')
            self.DATADICT[pulse]["PULSE_TIMES"] = RSPTIMES
        self.doneTrigger.emit() 
        self.updateProcTrigger.emit()  

    def computeWindow(self, pulse, band, centre, ftype, canvas=None):
        
        # Compute window
        nd = len(self.DATADICT[pulse][self.DATADICT[pulse]["chan"][0]][0][self.DATADICT["stacks"][0]]) # num. data 
        fft1 = np.fft.rfft(self.DATADICT[pulse][self.DATADICT[pulse]["chan"][0]][0][self.DATADICT["stacks"][0]])
        freqs   = np.fft.fftfreq(nd, self.dt)
        df      = freqs[1] - freqs[0]
        N       = int((round)(band/df))

        if ftype == "Hamming":
            window  = np.hamming(N)
        elif ftype == "Hanning":
            window  = np.hanning(N)
        elif ftype == "Rectangular":
            window  = np.ones(N)
        elif ftype == "Flat top":
            window = signal.flattop(N)
        else:
            print ("in windowFilter, window type undefined")

        WINDOW  = np.zeros(len(fft1))
        ifreq   = int(round(centre/df))
        istart =  ifreq-len(window)//2
        iend = 0
        if N%2:
            WINDOW[ifreq-N//2:ifreq+N//2+1] = window
            iend =  ifreq+N//2+1
        else:
            WINDOW[ifreq-N//2:ifreq+N//2] = window
            iend = ifreq+N//2
        
        self.WINDOW = WINDOW
        self.iWindowStart = istart
        self.iWindowEnd = iend
        self.FFTtimes = nd
        
        fft1 = np.fft.irfft(WINDOW)
        # calculate dead time 
        self.windead = 0.    
        for ift in np.arange(100,0,-1):
            #print( ift, fft1[ift] )
            if (abs(fft1[ift])/abs(fft1[0])) > 1e-2:
                #print ("DEAD TIME", 1e3*self.DATADICT[pulse]["TIMES"][ift] - 1e3*self.DATADICT[pulse]["TIMES"][0] ) 
                dead = 1e3*self.DATADICT[pulse]["TIMES"][ift] - 1e3*self.DATADICT[pulse]["TIMES"][0]
                self.windead = self.DATADICT[pulse]["TIMES"][ift] - self.DATADICT[pulse]["TIMES"][0]
                break

        if canvas != None:
            canvas.fig.clear()
            canvas.ax1  = canvas.fig.add_axes([.1, .6, .75, .35])
            canvas.ax2  = canvas.fig.add_axes([.1, .1, .75, .35])
            canvas.ax1.plot(WINDOW) 
            canvas.ax2.plot( 1e3* self.DATADICT[pulse]["TIMES"][0:100] - 1e3*self.DATADICT[pulse]["TIMES"][0], fft1[0:100] ) 
            canvas.ax2.set_xlabel("time (ms)")
            canvas.ax2.set_title("IFFT")
            canvas.draw()

        return [WINDOW, nd, istart, iend, dead, ift]

    def windowFilter(self, ftype, band, centre, trunc, canvas):

        ###############################
        # Window Filter (Ormsby filter http://www.xsgeo.com/course/filt.htm) 
        # apply window
        iFID = 0
        for pulse in self.DATADICT["PULSES"]:
            [WINDOW, nd, istart, iend, dead, idead] = self.computeWindow(pulse, band, centre, ftype)
            for istack in self.DATADICT["stacks"]:
                for ipm in range(self.DATADICT["nPulseMoments"]):
                    for ichan in np.append(self.DATADICT[pulse]["chan"], self.DATADICT[pulse]["rchan"]):
                        fft = np.fft.rfft( self.DATADICT[pulse][ichan][ipm][istack]  )
                        fft *= WINDOW
                        if trunc:
                            self.DATADICT[pulse][ichan][ipm][istack] = np.fft.irfft(fft, nd)[idead:-idead]
                        else:
                            self.DATADICT[pulse][ichan][ipm][istack] = np.fft.irfft(fft, nd)
                
                    percent = (int)(1e2*((float)(iFID*self.DATADICT["nPulseMoments"]+(ipm))/(len(self.DATADICT["PULSES"])*self.nPulseMoments)))
                    self.progressTrigger.emit(percent)  
                iFID += 1
            if trunc:
                self.DATADICT[pulse]["TIMES"] = self.DATADICT[pulse]["TIMES"][idead:-idead]
                [WINDOWxx, ndxx, istart, iend, deadxx, ideadxx] = self.computeWindow(pulse, band, centre, ftype)

            self.plotFT(canvas, istart, iend)
        self.doneTrigger.emit() 

    def bandpassFilter(self, canvas, blank, plot=True):

        if plot:
            canvas.reAx2()
            canvas.ax1.set_ylabel(r"signal [nV]", fontsize=8)
            canvas.ax2.set_xlabel(r"time [s]", fontsize=8)
            canvas.ax2.set_ylabel(r"signal [nV]", fontsize=8)

        ife = (int)( max(self.fe, self.windead) * self.samp )
        # Data
        iFID = 0
        for pulse in self.DATADICT["PULSES"]:
            self.DATADICT[pulse]["TIMES"] = self.DATADICT[pulse]["TIMES"][ife:-ife]
            for ipm in range(self.DATADICT["nPulseMoments"]):
                for istack in self.DATADICT["stacks"]:
                    if plot:
                        canvas.softClear()
                        mmax = 0
                        for ichan in self.DATADICT[pulse]["rchan"]:
                            canvas.ax1.plot( self.DATADICT[pulse]["TIMES"], 1e9*self.DATADICT[pulse][ichan][ipm][istack][ife:-ife], alpha=.5)
                            mmax = max( mmax, np.max(1e9*self.DATADICT[pulse][ichan][ipm][istack][ife:-ife])) 
                        for ichan in self.DATADICT[pulse]["chan"]:
                            canvas.ax2.plot( self.DATADICT[pulse]["TIMES"], 1e9*self.DATADICT[pulse][ichan][ipm][istack][ife:-ife], alpha=.5)
                            mmax = max( mmax, np.max(1e9*self.DATADICT[pulse][ichan][ipm][istack][ife:-ife])) 
                        canvas.ax2.set_prop_cycle(None)
                        canvas.ax1.set_prop_cycle(None)
                        canvas.ax1.set_ylim(-mmax, mmax) 

                    for ichan in self.DATADICT[pulse]["rchan"]:
                        # reflect signal back on itself to reduce gibbs effects on early times 
                        #nr = len( self.DATADICT[pulse][ichan][ipm][istack] ) - 1 + ife
                        #refl = np.append( -1*self.DATADICT[pulse][ichan][ipm][istack][::-1][0:-1], self.DATADICT[pulse][ichan][ipm][istack] )
                        #reflfilt = signal.filtfilt( self.filt_b, self.filt_a, refl )
                        #self.DATADICT[pulse][ichan][ipm][istack] = reflfilt[nr:-ife]
                        
                        # don't reflect
                        self.DATADICT[pulse][ichan][ipm][istack] = \
                            signal.filtfilt(self.filt_b, self.filt_a, self.DATADICT[pulse][ichan][ipm][istack])[ife:-ife]

                        # plot
                        if plot:
                            canvas.ax1.plot( self.DATADICT[pulse]["TIMES"], 1e9*self.DATADICT[pulse][ichan][ipm][istack], \
                                label = pulse + " ipm=" + str(ipm) + " istack=" + str(istack) + " rchan="  + str(ichan))

                    for ichan in self.DATADICT[pulse]["chan"]:
                        # reflect signal back on itself to reduce gibbs effects on early times 
                        #nr = len( self.DATADICT[pulse][ichan][ipm][istack] ) - 1 + ife
                        #refl = np.append( -1*self.DATADICT[pulse][ichan][ipm][istack][::-1][0:-1], self.DATADICT[pulse][ichan][ipm][istack] )
                        #reflfilt = signal.filtfilt( self.filt_b, self.filt_a, refl )
                        #self.DATADICT[pulse][ichan][ipm][istack] = reflfilt[nr:-ife]
                        
                        # don't reflect
                        self.DATADICT[pulse][ichan][ipm][istack] = \
                            scipy.signal.filtfilt(self.filt_b, self.filt_a, self.DATADICT[pulse][ichan][ipm][istack])[ife:-ife]
               
                        # plot
                        if plot:
                            canvas.ax2.plot( self.DATADICT[pulse]["TIMES"], 1e9*self.DATADICT[pulse][ichan][ipm][istack], \
                                label = "data " + pulse + " ipm=" + str(ipm) + " istack=" + str(istack) + " chan="  + str(ichan))

                    if plot:
                        canvas.ax1.legend(prop={'size':6}, loc='upper right')
                        canvas.ax2.legend(prop={'size':6}, loc='upper right')
                        canvas.draw() 
                    
                percent = (int)(1e2*((float)(iFID*self.DATADICT["nPulseMoments"]+(ipm))/(len(self.DATADICT["PULSES"])*self.nPulseMoments)))
                self.progressTrigger.emit(percent)  
            iFID += 1
        self.doneTrigger.emit() 
        self.updateProcTrigger.emit()  

    def loadGMRBinaryFID( self, rawfname, istack ):
        """ Reads a single binary GMR file and fills into DATADICT
        """

        #################################################################################
        # figure out key data indices
        # Pulse        
        nps  = (int)((self.prePulseDelay)*self.samp)
        npul   = (int)(self.pulseLength[0]*self.samp) #+ 100 

        # Data 
        nds  = nps+npul+(int)((self.deadTime)*self.samp);        # indice pulse 1 data starts 
        nd1 = (int)(1.*self.samp)                                # samples in first pulse

        invGain = 1./self.RxGain        
        invCGain = self.CurrentGain        

        pulse = "Pulse 1"
        chan = self.DATADICT[pulse]["chan"] 
        rchan = self.DATADICT[pulse]["rchan"] 
        
        rawFile = open( rawfname, 'rb')

        for ipm in range(self.nPulseMoments):
            buf1 = rawFile.read(4)
            buf2 = rawFile.read(4)
                
            N_chan = struct.unpack('>i', buf1 )[0]
            N_samp = struct.unpack('>i', buf2 )[0]
 
            T = N_samp * self.dt 
            TIMES = np.arange(0, T, self.dt) - .0002 # small offset in GMR DAQ?

            DATA = np.zeros([N_samp, N_chan+1])
            for ichan in range(N_chan):
                DATADUMP = rawFile.read(4*N_samp)
                for irec in range(N_samp):
                    DATA[irec,ichan] = struct.unpack('>f', DATADUMP[irec*4:irec*4+4])[0]
                           
            # Save into Data Cube 
            for ichan in chan:
                self.DATADICT["Pulse 1"][ichan][ipm][istack] = DATA[:,eval(ichan)+3][nds:nds+nd1] * invGain 
                self.DATADICT["Pulse 1"]["TIMES"] = TIMES[nds:nds+nd1]
                self.DATADICT["Pulse 1"]["CURRENT"][ipm][istack] = DATA[:,1][nps:nps+npul] * invCGain
                self.DATADICT["Pulse 1"]["PULSE_TIMES"] = TIMES[nps:nps+npul] 

            # reference channels?
            for ichan in rchan:
                self.DATADICT["Pulse 1"][ichan][ipm][istack] = DATA[:,eval(ichan)+3][nds:nds+nd1] * invGain 
                self.DATADICT["Pulse 1"]["TIMES"] = TIMES[nds:nds+nd1]
    
    def loadGMRASCIIFID( self, rawfname, istack ):
        """Based on the geoMRI instrument manufactured by VistaClara. Imports 
        a suite of raw .lvm files with the following format (on one line)

        time(s) DC_Bus/100(V) Current+/75(A)  Curr-/75(A)  Voltage+/200(V) \  
        Ch1(V) Ch2(V) Ch3(V) Ch4(V)

        Sampling rate is assumed at 50 kHz 
        """
        import pandas as pd 
        #################################################################################
        # figure out key data indices
        # Pulse        
        nps  = (int)((self.prePulseDelay)*self.samp)
        npul   = (int)(self.pulseLength[0]*self.samp) #+ 100 

        # Data 
        nds  = nps+npul+(int)((self.deadTime)*self.samp);        # indice pulse 1 data starts 
        nd1 = (int)(1.*self.samp) - nds                          # samples in first pulse
        ndr = (int)(1.*self.samp)                                # samples in record 

        invGain = 1./self.RxGain        
        invCGain = self.CurrentGain        

        pulse = "Pulse 1"
        chan = self.DATADICT[pulse]["chan"] 
        rchan = self.DATADICT[pulse]["rchan"] 
            
        T = 1.5 #N_samp * self.dt 
        TIMES = np.arange(0, T, self.dt) - .0002 # small offset in GMR DAQ?
        
        self.DATADICT["Pulse 1"]["TIMES"] = TIMES[nds:nds+nd1]
        self.DATADICT["Pulse 1"]["PULSE_TIMES"] = TIMES[nps:nps+npul]

        # pandas is much faster than numpy for io
        #DATA = np.loadtxt(rawfname)
        DATA = pd.read_csv(rawfname, header=None, sep="\t").values
        for ipm in range(self.nPulseMoments):
            for ichan in np.append(chan,rchan):
                self.DATADICT["Pulse 1"][ichan][ipm][istack] =  DATA[:, eval(ichan)+4][nds:(nds+nd1)] * invGain
                self.DATADICT["Pulse 1"]["CURRENT"][ipm][istack] = DATA[:,2][nps:nps+npul] * invCGain
            nds += ndr
            nps += ndr 

    def loadGMRASCIIT1( self, rawfname, istack ):
        """Based on the geoMRI instrument manufactured by VistaClara. Imports 
        a suite of raw .lvm files with the following format (on one line)

        time(s) DC_Bus/100(V) Current+/75(A)  Curr-/75(A)  Voltage+/200(V) \  
        Ch1(V) Ch2(V) Ch3(V) Ch4(V)

        Sampling rate is assumed at 50 kHz 
        """
        import pandas as pd 
        #################################################################################
        # figure out key data indices
        # Pulse       
        nps  = (int)((self.prePulseDelay)*self.samp)
        npul = (int)(self.pulseLength[0]*self.samp) #+ 100 

        # phase cycling 
        # Older T1 GMR data had a curious phase cycling
        npc = 2 #(int)( self.samp / self.transFreq / 6 )
        #print("npc", npc)

        # Data 
        nds  = nps+npul+(int)((self.deadTime)*self.samp);           # indice pulse 1 data starts 
        nd1 = (int)( (self.interpulseDelay) * self.samp) - nds      # samples in first pulse
        ndr = (int)( (self.interpulseDelay) * self.samp)            # samples in record 

        invGain = 1./self.RxGain        
        invCGain = self.CurrentGain        

        pulse = "Pulse 1"
        chan = self.DATADICT[pulse]["chan"] 
        rchan = self.DATADICT[pulse]["rchan"] 
            
        T = 1.5 #N_samp * self.dt 
        TIMES = np.arange(0, T, self.dt) - .0002 # small offset in GMR DAQ?
        
        self.DATADICT["Pulse 1"]["TIMES"]       = TIMES[nds:nds+nd1]
        self.DATADICT["Pulse 1"]["PULSE_TIMES"] = TIMES[nps:nps+npul]

        # pandas is much faster than numpy for io
        #DATA = np.loadtxt(rawfname)
        DATA = pd.read_csv(rawfname, header=None, sep="\t").values
        for ipm in range(self.nPulseMoments):
            for ichan in np.append(chan,rchan):
                if ipm%2:
                    self.DATADICT["Pulse 1"][ichan][ipm][istack]     =  DATA[:, eval(ichan)+4][(nds+npc):(nds+nd1+npc)] * invGain
                    #self.DATADICT["Pulse 1"][ichan][ipm][istack]     =  DATA[:, eval(ichan)+4][nds:(nds+nd1)] * invGain
                    self.DATADICT["Pulse 1"]["CURRENT"][ipm][istack] =  DATA[:,2][nps+npc:nps+npul+npc] * invCGain
                else:
                    self.DATADICT["Pulse 1"][ichan][ipm][istack]     =  DATA[:, eval(ichan)+4][nds:(nds+nd1)] * invGain
                    self.DATADICT["Pulse 1"]["CURRENT"][ipm][istack] =  DATA[:,2][nps:nps+npul] * invCGain
            nds += ndr
            nps += ndr 

    def loadFIDData(self, base, procStacks, chanin, rchanin, FIDProc, canvas, deadTime, plot):
        '''
            Loads a GMR FID dataset, reads binary and ASCII format files 
        '''

        canvas.reAx3(True,False)

        chan = []
        for ch in chanin:
            chan.append(str(ch)) 
        
        rchan = []
        for ch in rchanin:
            rchan.append(str(ch)) 

        self.deadTime = deadTime             # instrument dead time before measurement
        self.samp = 50000.                   # in case this is a reproc, these might have 
        self.dt   = 1./self.samp             # changed


        #################################################################################
        # Data structures     
        PULSES = [FIDProc]
        PULSES = ["Pulse 1"]

        self.DATADICT = {}
        self.DATADICT["nPulseMoments"] = self.nPulseMoments
        self.DATADICT["stacks"] = procStacks
        self.DATADICT["PULSES"] = PULSES
        for pulse in PULSES: 
            self.DATADICT[pulse] = {}
            self.DATADICT[pulse]["chan"] = chan        # TODO these should not be a subet of pulse! for GMR all 
            self.DATADICT[pulse]["rchan"] = rchan      #      data are consistent 
            self.DATADICT[pulse]["CURRENT"] = {} 
            for ichan in np.append(chan,rchan):
                self.DATADICT[pulse][ichan] = {}
                for ipm in range(self.nPulseMoments):
                    self.DATADICT[pulse][ichan][ipm] = {} 
                    self.DATADICT[pulse]["CURRENT"][ipm] = {} 
                    for istack in procStacks:
                        self.DATADICT[pulse][ichan][ipm][istack] = np.zeros(3)
                        self.DATADICT[pulse]["CURRENT"][ipm][istack] = np.zeros(3) 

        ##############################################
        # Read in binary (.lvm) data
        iistack = 0
        for istack in procStacks:
            if self.nDAQVersion <= 1.0:
                try:
                    self.loadGMRASCIIFID( base + "_" + str(istack), istack )
                except:
                    self.loadGMRASCIIFID( base + "_" + str(istack) + ".lvm", istack )
            elif self.nDAQVersion < 2.3:
                self.loadGMRASCIIFID( base + "_" + str(istack), istack )
            else:
                self.loadGMRBinaryFID( base + "_" + str(istack) + ".lvm", istack )

            if plot: 
                for ipm in range(self.nPulseMoments):
                    canvas.softClear()                           

                    for ichan in chan:
                        canvas.ax1.plot(self.DATADICT["Pulse 1"]["PULSE_TIMES"], self.DATADICT["Pulse 1"]["CURRENT"][ipm][istack] , color='black')
                        canvas.ax3.plot(self.DATADICT["Pulse 1"]["TIMES"],       self.DATADICT["Pulse 1"][ichan][ipm][istack], label="Pulse 1 FID data ch. "+str(ichan)) #, color='blue')

                    for ichan in rchan:
                        canvas.ax2.plot(self.DATADICT["Pulse 1"]["TIMES"], self.DATADICT["Pulse 1"][ichan][ipm][istack], label="Pulse 1 FID ref ch. "+str(ichan)) #, color='blue')

                    # reference axis
                    canvas.ax2.tick_params(axis='both', which='major', labelsize=10)
                    canvas.ax2.tick_params(axis='both', which='minor', labelsize=10)
                    #canvas.ax2.xaxis.set_ticklabels([])
                    plt.setp(canvas.ax2.get_xticklabels(), visible=False)
                    canvas.ax2.legend(prop={'size':10}, loc='upper right')
                    canvas.ax2.set_title("stack "+str(istack)+" pulse index " + str(ipm), fontsize=10)
                    canvas.ax2.set_ylabel("RAW signal [V]", fontsize=10)


                    canvas.ax1.set_ylabel("Current (A)", fontsize=10) 
                    canvas.ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y') 
                    canvas.ax1.set_xlabel("time (s)", fontsize=10)

                    canvas.ax3.legend(prop={'size':10}, loc='upper right')
                    canvas.ax3.set_ylabel("RAW signal [V]", fontsize=10)
                    
                    canvas.fig.tight_layout()
                    canvas.draw()

            percent = (int) (1e2*((float)((iistack*self.nPulseMoments+ipm+1))  / (len(procStacks)*self.nPulseMoments)))
            self.progressTrigger.emit(percent) 
            iistack += 1

#                percent = (int) (1e2*((float)((iistack*self.nPulseMoments+ipm+1))  / (len(procStacks)*self.nPulseMoments)))
#                self.progressTrigger.emit(percent) 
#                iistack += 1

        self.enableDSP()    
        self.doneTrigger.emit()

    def loadT1Data(self, base, procStacks, chanin, rchanin, FIDProc, canvas, deadTime, plot):
        '''
            Loads a GMR T1 dataset, reads binary and ASCII format files 
        '''

        canvas.reAx3(True,False)

        chan = []
        for ch in chanin:
            chan.append(str(ch)) 
        
        rchan = []
        for ch in rchanin:
            rchan.append(str(ch)) 

        # not in any headers but this has changed, NOT the place to do this. MOVE  
        #self.prePulseDelay  = 0.01          # delay before pulse
        self.deadTime       = deadTime       # instrument dead time before measurement
        self.samp = 50000.                   # in case this is a reproc, these might have 
        self.dt   = 1./self.samp             # changed


        #################################################################################
        # Data structures     
        PULSES = [FIDProc]

        self.DATADICT = {}
        self.DATADICT["nPulseMoments"] = self.nPulseMoments
        self.DATADICT["stacks"] = procStacks
        self.DATADICT["PULSES"] = PULSES
        for pulse in PULSES: 
            self.DATADICT[pulse] = {}
            self.DATADICT[pulse]["chan"] = chan        # TODO these should not be a subet of pulse! for GMR all 
            self.DATADICT[pulse]["rchan"] = rchan      #      data are consistent 
            self.DATADICT[pulse]["CURRENT"] = {} 
            for ichan in np.append(chan,rchan):
                self.DATADICT[pulse][ichan] = {}
                for ipm in range(self.nPulseMoments):
                    self.DATADICT[pulse][ichan][ipm] = {} 
                    self.DATADICT[pulse]["CURRENT"][ipm] = {} 
                    for istack in procStacks:
                        self.DATADICT[pulse][ichan][ipm][istack] = np.zeros(3)
                        self.DATADICT[pulse]["CURRENT"][ipm][istack] = np.zeros(3) 

        ##############################################
        # Read in binary (.lvm) data
        iistack = 0
        fnames = []
        for istack in procStacks:
            if self.nDAQVersion < 2.3:
                #rawfname = base + "_" + str(istack) 
                #self.loadGMRASCIIFID( base + "_" + str(istack), istack )
                self.loadGMRASCIIT1( base + "_" + str(istack), istack )
            else:
                self.loadGMRBinaryFID( base + "_" + str(istack) + ".lvm", istack )
                #fnames.append( base + "_" + str(istack) + ".lvm" )
                
            percent = (int) (1e2*((float)((iistack*self.nPulseMoments+ipm+1))  / (len(procStacks)*self.nPulseMoments)))
            self.progressTrigger.emit(percent) 
            iistack += 1

        # multiprocessing load data
        #info = {}
        #info["prePulseDelay"] = self.prePulseDelay
        #info["samp"] = self.samp
        #with multiprocessing.Pool() as pool: 
        #    results = pool.starmap( xxloadGMRBinaryFID, ( fnames, zip(itertools.repeat(info)) ) ) 
        # Plotting

        if plot: 
            iistack = 0
            for istack in procStacks:
                #for ipm in range(0,7,1):
                for ipm in range(self.nPulseMoments):
                    canvas.ax1.clear()
                    canvas.ax2.clear()
                    canvas.ax3.clear()
                    #canvas.fig.patch.set_facecolor('blue')
                    for ichan in chan:
                        canvas.ax1.plot(self.DATADICT["Pulse 1"]["PULSE_TIMES"], self.DATADICT["Pulse 1"]["CURRENT"][ipm][istack] , color='black')
                        canvas.ax3.plot(self.DATADICT["Pulse 1"]["TIMES"],       self.DATADICT["Pulse 1"][ichan][ipm][istack], label="Pulse 1 FID data ch. "+str(ichan)) #, color='blue')

                    for ichan in rchan:
                        canvas.ax2.plot(self.DATADICT["Pulse 1"]["TIMES"], self.DATADICT["Pulse 1"][ichan][ipm][istack], label="Pulse 1 FID ref ch. "+str(ichan)) #, color='blue')

                    canvas.ax3.legend(prop={'size':6}, loc='upper right')
                    canvas.ax2.legend(prop={'size':6}, loc='upper right')
                    
                    canvas.ax1.set_title("stack "+str(istack)+" pulse index " + str(ipm), fontsize=8)
                    canvas.ax1.set_xlabel("time [s]", fontsize=8)
                    canvas.ax1.set_ylabel("Current [A]", fontsize=8) 
                    canvas.ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y') 
                    
                    canvas.ax2.set_ylabel("RAW signal [V]", fontsize=8)
                    canvas.ax2.tick_params(axis='both', which='major', labelsize=8)
                    canvas.ax2.tick_params(axis='both', which='minor', labelsize=6)
                    canvas.ax2.set_xlabel("time [s]", fontsize=8)
                    canvas.ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y') 
                    canvas.ax3.ticklabel_format(style='sci', scilimits=(0,0), axis='y') 
                    canvas.draw()
                #canvas.draw()

                percent = (int) (1e2*((float)((iistack*self.nPulseMoments+ipm+1))  / (len(procStacks)*self.nPulseMoments)))
                self.progressTrigger.emit(percent) 
                iistack += 1

        self.enableDSP()    
        self.doneTrigger.emit()   
 
    def load4PhaseT1Data(self, base, procStacks, chan, rchan, FIDProc, canvas, deadTime, plot): 

        """
            Designed to load GMR 4-phase data which use the following convention for phase cycles
                         P1     P2
            Stack 1 ->   0       0     <--   <--   
            Stack 2 ->   0      pi/2     |   <--  <--
            Stack 3 ->   pi/2    0     <--          |  <--
            Stack 4 ->   pi/2   pi/2              <--  <--
            The cycle is determined by stack indice. Walbrecker proposes for pulse2 data  (Stack2 - Stack1) / 2
                equivalently (Stack 4 - Stack3) will yield the same voltage response wrt. the second pulse. 
            Alternatively Stack 4 can be converted to be aligned with Stack 1 by negating, and Stack 3 Can be aligned with Stack 2 by negating
            Then there are just the two phase cycles that can be stacked like normal.
            Unfortunately, we need to stack each cycle first, then perform corrections for phase cycling. The reason for this is that otherwise, 
            the entire point is lost, as the signal that is desired to be cancelled out may not be balanced evenly across the stacks. That is to say, 
            if there is an uneven number of a certain phase cycle. 

            We could, I suppose impose this condition, but I think I would rather not? 
                + more samples for std. deviation calculation 
                + single spikes will have less residual effect
                - can no longer do normality tests etc. and remove data that are suspect.
                - requires a dumb stack, and may also require removal of entire stacks of data  

            Additonally, the GMR varies phase as a function of pulse moment index, so that the first pusle moment is zero phase, the second is pi/2 the third is zero ... 
            This however, is altered by the above convention. It gets a little complicated...
        """

        import struct
        canvas.reAx3()

        # not in any headers but this has changed, NOT the place to do this. MOVE  
        self.prePulseDelay  = 0.01           # delay before pulse
        self.deadTime       = deadTime       # instrument dead time before measurement
        self.samp = 50000.                   # in case this is a reproc, these might have 
        self.dt   = 1./self.samp             # changed
        invGain = 1./self.RxGain        
        invCGain = self.CurrentGain        

        #################################################################################
        # figure out key data indices
        # Pulse        
        nps  = (int)((self.prePulseDelay)*self.samp)
        nps2 = (int)((self.prePulseDelay+self.interpulseDelay)*self.samp)
        npul   = (int)(self.pulseLength[0]*self.samp) #+ 100 
        np2  = (int)(self.pulseLength[1]*self.samp) #+ 100

        # Data 
        nds  = nps+npul+(int)((self.deadTime)*self.samp);        # indice pulse 1 data starts 
        nd1 = (int)((self.interpulseDelay)*self.samp)      # samples in first pulse
        
        nd2s = nps+npul+nd1+(int)((self.deadTime)*self.samp);   # indice pulse 2 data starts 
        nd2 =  (int)((1.)*self.samp)                          # samples in first pulse
        nd1 -= (int)((.028)*self.samp) + nps   # some time to get ready for next pulse

        #################################################################################
        # Data structures     
        PULSES = [FIDProc]
        if FIDProc == "Both":
            PULSES = ["Pulse 1","Pulse 2"]

        self.DATADICT = {}
        self.DATADICT["nPulseMoments"] = self.nPulseMoments
        self.DATADICT["stacks"] = procStacks
        self.DATADICT["PULSES"] = PULSES
        for pulse in PULSES: 
            self.DATADICT[pulse] = {}
            self.DATADICT[pulse]["chan"] = chan
            self.DATADICT[pulse]["rchan"] = rchan
            self.DATADICT[pulse]["CURRENT"] = {} 
            for ichan in np.append(chan,rchan):
                self.DATADICT[pulse][ichan] = {}
                for ipm in range(self.nPulseMoments):
                    self.DATADICT[pulse][ichan][ipm] = {} 
                    self.DATADICT[pulse]["CURRENT"][ipm] = {} 
                    for istack in procStacks:
                        self.DATADICT[pulse][ichan][ipm][istack] = np.zeros(3)
                        self.DATADICT[pulse]["CURRENT"][ipm][istack] = np.zeros(3) 

        ##############################################
        # Read in binary data
        iistack = 0
        for istack in procStacks:
            rawFile = open(base + "_" + str(istack) + ".lvm", 'rb')
            for ipm in range(self.nPulseMoments):

                N_chan = struct.unpack('>i', rawFile.read(4))[0]
                N_samp = struct.unpack('>i', rawFile.read(4))[0]
                
                T = N_samp * self.dt 
                TIMES = np.arange(0, T, self.dt) - .0002 # small offset in GMR DAQ?

                DATA = np.zeros([N_samp, N_chan+1])
                for ichan in range(N_chan):
                    DATADUMP = rawFile.read(4*N_samp)
                    for irec in range(N_samp):
                        DATA[irec,ichan] = struct.unpack('>f', DATADUMP[irec*4:irec*4+4])[0]
                if plot: 
                    #canvas.ax1.clear()
                    #canvas.ax2.clear()
                    canvas.softClear()
                li = np.shape( DATA[:,4][nd2s:nd2s+nd2] )[0]               

                ######################################
                # save into DATA cube
                # TODO, changing iFID to 'Pulse 1' or 'Pulse 2'
                for ichan in chan:
                    if FIDProc == "Pulse 1":
                        self.DATADICT["Pulse 1"][ichan][ipm][istack] = DATA[:,ichan+3][nds:nds+nd1] * invGain 
                        self.DATADICT["Pulse 1"]["TIMES"] = TIMES[nds:nds+nd1]
                        self.DATADICT["Pulse 1"]["CURRENT"][ipm][istack] = DATA[:,1][nps:nps+npul] * invCGain
                        self.DATADICT["Pulse 1"]["PULSE_TIMES"] = TIMES[nps:nps+npul] 
                        if plot:
                            canvas.ax3.plot(self.DATADICT["Pulse 1"]["TIMES"],       self.DATADICT["Pulse 1"][ichan][ipm][istack], label="Pulse 1 FID data ch. "+str(ichan)) #, color='blue')
                            canvas.ax1.plot(self.DATADICT["Pulse 1"]["PULSE_TIMES"], self.DATADICT["Pulse 1"]["CURRENT"][ipm][istack] , color='black')
                    elif FIDProc == "Pulse 2":
                        print("TODO fix y scale")
                        self.DATADICT["Pulse 2"][ichan][ipm][istack] = DATA[:,ichan+3][nd2s:nd2s+nd2] *invGain
                        self.DATADICT["Pulse 2"]["TIMES"] = TIMES[nd2s:nd2s+nd2]
                        self.DATADICT["Pulse 2"]["CURRENT"][ipm][istack] = DATA[:,1][nps2:nps2+np2] * invCGain
                        self.DATADICT["Pulse 2"]["PULSE_TIMES"] = TIMES[nps2:nps2+np2] 
                        if plot:
                            canvas.ax3.plot(self.DATADICT["Pulse 2"]["TIMES"], self.DATADICT["Pulse 2"][ichan][ipm][istack], label="Pulse 2 FID data ch. "+str(ichan)) #, color='blue')
                            canvas.ax1.plot( self.DATADICT["Pulse 2"]["PULSE_TIMES"], self.DATADICT["Pulse 2"]["CURRENT"][ipm][istack], color='black' )
                    else:
                        self.DATADICT["Pulse 1"][ichan][ipm][istack] = DATA[:,ichan+3][nds:nds+nd1] * invGain
                        self.DATADICT["Pulse 2"][ichan][ipm][istack] = DATA[:,ichan+3][nd2s:nd2s+nd2] * invGain
                        self.DATADICT["Pulse 1"]["TIMES"] = TIMES[nds:nds+nd1]
                        self.DATADICT["Pulse 2"]["TIMES"] = TIMES[nd2s:nd2s+nd2]
                        self.DATADICT["Pulse 1"]["CURRENT"][ipm][istack] = DATA[:,1][nps:nps+npul] * invCGain 
                        self.DATADICT["Pulse 1"]["PULSE_TIMES"] = TIMES[nps:nps+npul] 
                        self.DATADICT["Pulse 2"]["CURRENT"][ipm][istack] = DATA[:,1][nps2:nps2+np2] * invCGain
                        self.DATADICT["Pulse 2"]["PULSE_TIMES"] = TIMES[nps2:nps2+np2] 
                        if plot:
                            canvas.ax3.plot(self.DATADICT["Pulse 1"]["TIMES"], self.DATADICT["Pulse 1"][ichan][ipm][istack], label="Pulse 1 FID data ch. "+str(ichan)) #, color='blue')
                            canvas.ax3.plot(self.DATADICT["Pulse 2"]["TIMES"], self.DATADICT["Pulse 2"][ichan][ipm][istack], label="Pulse 2 FID data ch. "+str(ichan)) #, color='blue')
                            canvas.ax1.plot( self.DATADICT["Pulse 1"]["PULSE_TIMES"], self.DATADICT["Pulse 1"]["CURRENT"][ipm][istack] , color='black' )
                            canvas.ax1.plot( self.DATADICT["Pulse 2"]["PULSE_TIMES"], self.DATADICT["Pulse 2"]["CURRENT"][ipm][istack] , color='black')
                
                for ichan in rchan:
                    if FIDProc == "Pulse 1":
                        self.DATADICT["Pulse 1"][ichan][ipm][istack] = DATA[:,ichan+3][nds:nds+nd1] * invGain 
                        self.DATADICT["Pulse 1"]["TIMES"] = TIMES[nds:nds+nd1]
                        if plot:
                            canvas.ax2.plot(self.DATADICT["Pulse 1"]["TIMES"], self.DATADICT["Pulse 1"][ichan][ipm][istack], label="Pulse 1 FID ref ch. "+str(ichan)) #, color='blue')
                    elif FIDProc == "Pulse 2":
                        self.DATADICT["Pulse 2"][ichan][ipm][istack] = DATA[:,ichan+3][nd2s:nd2s+nd2] * invGain
                        self.DATADICT["Pulse 2"]["TIMES"] = TIMES[nd2s:nd2s+nd2]
                        if plot:
                            canvas.ax2.plot(self.DATADICT["Pulse 2"]["TIMES"], self.DATADICT["Pulse 2"][ichan][ipm][istack], label="Pulse 2 FID ref ch. "+str(ichan)) #, color='blue')
                    else:
                        self.DATADICT["Pulse 1"][ichan][ipm][istack] = DATA[:,ichan+3][nds:nds+nd1] * invGain
                        self.DATADICT["Pulse 2"][ichan][ipm][istack] = DATA[:,ichan+3][nd2s:nd2s+nd2] * invGain
                        self.DATADICT["Pulse 1"]["TIMES"] = TIMES[nds:nds+nd1]
                        self.DATADICT["Pulse 2"]["TIMES"] = TIMES[nd2s:nd2s+nd2]
                        if plot:
                            canvas.ax2.plot(self.DATADICT["Pulse 1"]["TIMES"], self.DATADICT["Pulse 1"][ichan][ipm][istack], label="Pulse 1 FID ref ch. "+str(ichan)) #, color='blue')
                            canvas.ax2.plot(self.DATADICT["Pulse 2"]["TIMES"], self.DATADICT["Pulse 2"][ichan][ipm][istack], label="Pulse 2 FID ref ch. "+str(ichan)) #, color='blue')
                
                if plot:
                    canvas.ax3.legend(prop={'size':6}, loc='upper right')
                    canvas.ax2.legend(prop={'size':6}, loc='upper right')
                    canvas.ax1.set_title("stack "+str(istack)+" pulse index " + str(ipm), fontsize=8)
                    canvas.ax1.set_xlabel("time [s]", fontsize=8)
                    canvas.ax3.set_ylabel("RAW signal [V]", fontsize=8)
                    canvas.ax2.set_ylabel("RAW signal [V]", fontsize=8)
                    canvas.ax1.set_ylabel("Current [A]", fontsize=8) 
                    #canvas.ax2.tick_params(axis='both', which='major', labelsize=8)
                    #canvas.ax2.tick_params(axis='both', which='minor', labelsize=6)
                    #canvas.ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y') 
                    #canvas.ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y') 

                    canvas.draw()

                # update GUI of where we are
                percent = (int) (1e2*((float)((iistack*self.nPulseMoments+ipm+1))  / (len(procStacks)*self.nPulseMoments)))
                self.progressTrigger.emit(percent)  
            iistack += 1
        
        self.enableDSP()    
        self.doneTrigger.emit() 


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print( "mrsurvey path/to/header   <stack1>   <stackN> ")
        exit()

    GMR = GMRDataProcessor() 
    GMR.readHeaderFile(sys.argv[1])
    GMR.Print()

    if GMR.pulseType == "FID":
        GMR.loadFIDData(sys.argv[1], sys.argv[2], sys.argv[3], 5)
    
    if GMR.pulseType == "4PhaseT1":
        GMR.load4PhaseT1Data(sys.argv[1], sys.argv[2], sys.argv[3], 5)

    pylab.show()

