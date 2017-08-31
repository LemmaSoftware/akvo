from SEGPlot import *

import sys
sys.path.append( '../tressel' )

import matplotlib.pyplot as plt
import matplotlib.ticker
import scipy.io as sio
import scipy.signal as signal
import numpy as np

import mrsurvey
import pickle
import decay
import cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='inferno', cmap=cmaps.inferno)
plt.register_cmap(name='inferno_r', cmap=cmaps.inferno_r)

plt.register_cmap(name='magma', cmap=cmaps.magma)
plt.register_cmap(name='magma_r', cmap=cmaps.magma_r)


class canvas():
    
    def __init__(self):
        self.fig = plt.figure( figsize=(pc2in(20),pc2in(20) ) )
        self.ax1 = self.fig.add_subplot((211))
        self.ax2 = self.fig.add_subplot((212), sharex=self.ax1)
   
    def draw(self):
        plt.draw() 

    def reAx2(self):
        try:
            self.fig.clear()
        except:
            pass

        try:
            self.ax1.clear() 
            self.delaxes(self.ax1) #.clear()
        except:
            pass
            
        try:
            self.ax2.clear() 
            self.delaxes(self.ax2) #.clear()
        except:
            pass

        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)

        self.ax1.tick_params(axis='both', which='major', labelsize=8)
        self.ax2.tick_params(axis='both', which='major', labelsize=8)

        self.ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
        self.ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  

        self.ax1.yaxis.get_offset_text().set_size(8) 
        self.ax2.yaxis.get_offset_text().set_size(8) 

    def reAx4(self):
        try:
            self.fig.clear()
        except:
            pass

        # two main axes
        self.ax1 = self.fig.add_axes([0.15, 0.55,   0.625, 0.3672])
        self.ax2 = self.fig.add_axes([0.15, 0.135 , 0.625, 0.3672])
        
        # for colourbars
        self.cax1 = self.fig.add_axes([0.8, 0.55 ,  0.025, 0.3672])
        self.cax2 = self.fig.add_axes([0.8, 0.135,  0.025, 0.3672])
        
        self.ax1.tick_params(axis='both', which='major', labelsize=8)
        self.ax2.tick_params(axis='both', which='major', labelsize=8)

        self.ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
        self.ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  

        self.ax1.yaxis.get_offset_text().set_size(8) 
        self.ax2.yaxis.get_offset_text().set_size(8) 

        self.cax1.tick_params(axis='both', which='major', labelsize=8)
        self.cax2.tick_params(axis='both', which='major', labelsize=8)

        self.cax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
        self.cax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  

        self.cax1.yaxis.get_offset_text().set_size(8) #.get_text()
        self.cax2.yaxis.get_offset_text().set_size(8) #.get_text()

        self.cax1.tick_params(labelsize=8) 
        self.cax2.tick_params(labelsize=8)
        
        #self.ax1.yaxis.minorticks_off()
        #self.ax2.yaxis.minorticks_off()
        
        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='minor',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') 

if __name__ == "__main__":
   
    first = True
    for ffile in sys.argv[1::]: 
        Canvas = canvas()
        pfile = file(ffile)
        unpickle = pickle.Unpickler(pfile)
        MRS = mrsurvey.GMRDataProcessor()
        MRS.DATADICT = unpickle.load()
        MRS.pulseType = MRS.DATADICT["INFO"]["pulseType"]
        MRS.transFreq = MRS.DATADICT["INFO"]["transFreq"]
        MRS.pulseLength = MRS.DATADICT["INFO"]["pulseLength"]
        MRS.TuneCapacitance = MRS.DATADICT["INFO"]["TuneCapacitance"]
        MRS.samp = MRS.DATADICT["INFO"]["samp"]
        MRS.nPulseMoments = MRS.DATADICT["INFO"]["nPulseMoments"]
        MRS.deadTime = MRS.DATADICT["INFO"]["deadTime"]

        MRS.quadDet(1, True, Canvas)
        MRS.gateIntegrate(14, 1, Canvas )

        #Canvas.fig.suptitle(r"\textbf{Experiment 0, channel 4}",  fontsize=8) #, fontweight='bold')
        Canvas.ax1.set_title(r"Experiment 0, channel 4",  fontsize=8) 
        #Canvas.ax1.set_title("Channel 4")



        plt.savefig("test.eps", dpi=2200)

        if first:
            mat = MRS.DATADICT["CA"]
            pmat = MRS.DATADICT["CP"]
            first = False
        else:
            mat += MRS.DATADICT["CA"]
            pmat += MRS.DATADICT["CP"]

    quadSum = True
    if quadSum:
    
        Canvas.ax1.clear() 
        Canvas.ax2.clear() 
        Canvas.cax1.clear() 
        Canvas.cax2.clear() 
        pulse = "Pulse 1"
        clip = 1
        QQ = np.average(MRS.DATADICT[pulse]["Q"], axis=1 )
        im1 = Canvas.ax1.pcolormesh( 1e3*MRS.DATADICT[pulse]["TIMES"][clip-1:-clip], QQ, mat,  cmap='coolwarm_r', rasterized=True, vmin=-np.max(np.abs(mat)), vmax=np.max(np.abs(mat)))
        im2 = Canvas.ax2.pcolormesh( 1e3*MRS.DATADICT[pulse]["TIMES"][clip-1:-clip], QQ, pmat, cmap='coolwarm_r', rasterized=True, vmin=-np.max(np.abs(pmat)), vmax=np.max(np.abs(pmat)))

        cb2 = Canvas.fig.colorbar(im2, cax=Canvas.cax2)
        cb2.set_label("Noise residual (nV)", fontsize=8)


        #canvas.ax2.yaxis.set_ticks( QQ[0,9::7] )       
        
        Canvas.ax1.set_yscale('log')
        Canvas.ax2.set_yscale('log')
        
        qlabs = np.append(np.concatenate( (QQ[0:1],QQ[9::10] )), QQ[-1] ) 
        Canvas.ax1.yaxis.set_ticks( qlabs ) # np.append(np.concatenate( (QQ[0:1],QQ[9::10] )), QQ[-1] ) )
        Canvas.ax2.yaxis.set_ticks( qlabs ) #np.append(np.concatenate( (QQ[0:1],QQ[9::10] )), QQ[-1] ) )
        #formatter = matplotlib.ticker.LogFormatter(10, labelOnlyBase=False)
        formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: str((round(x,1)))) 
        Canvas.ax1.yaxis.set_major_formatter(formatter)#matplotlib.ticker.FormatStrFormatter('%d.1'))
        Canvas.ax2.yaxis.set_major_formatter(formatter)#matplotlib.ticker.FormatStrFormatter('%d.1')) 

        plt.setp(Canvas.ax1.get_xticklabels(), visible=False)
 
        t = 1e3*MRS.DATADICT[pulse]["TIMES"][clip-1:-clip],
        Canvas.ax1.set_ylim( np.min(QQ), np.max(QQ) )
        Canvas.ax2.set_ylim( np.min(QQ), np.max(QQ) )
        Canvas.ax1.set_xlim( np.min(t), np.max(t) )
        Canvas.ax2.set_xlim( np.min(t), np.max(t) )

        cb1 = Canvas.fig.colorbar(im1, cax=Canvas.cax1)
        cb1.set_label("Phased amplitude (nV)", fontsize=8)

        Canvas.ax2.set_xlabel(r"Time (ms)", fontsize=8)
        Canvas.ax1.set_ylabel(r"$q$ ( $\mathrm{A}\cdot\mathrm{s}$)", fontsize=8)
        Canvas.ax2.set_ylabel(r"$q$ ( $\mathrm{A}\cdot\mathrm{s}$)", fontsize=8)

        plt.savefig("quadSum.eps")

    plt.show()
