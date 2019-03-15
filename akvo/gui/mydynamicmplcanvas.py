from __future__ import unicode_literals
import sys
import os
import random
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets

from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class MyMplCanvas(FigureCanvas):
    
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=3, height=4, dpi=100):
        
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='darkgrey') # this fucking works...why?
        #self.fig.patch.set_facecolor('blue')
        FigureCanvas.__init__(self, self.fig)

        self.setParent(parent)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

    def clicked(self):
        print ("Clicked")

class MyDynamicMplCanvas(MyMplCanvas):

    """A canvas that updates itself every second with a new plot."""
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.ax1 = self.fig.add_axes([.125,.1,.725,.8])
        self.ax2 = self.ax1.twinx() # fig.add_axes([.125,.1,.725,.8])
        self.compute_initial_figure()

    def reAxH(self, num, shx=True, shy=True):
        
        try:
            for ax in fig.axes:
                self.fig.delaxes(ax)
        except:
            pass
        try:
            self.fig.clear()
        except:
            pass

        for n in range(num):
            if n == 0:
                self.ax1 = self.fig.add_subplot( 1, num, 1)
                self.ax1.tick_params(axis='both', which='major', labelsize=8)
                self.ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='x')  
                self.ax1.yaxis.get_offset_text().set_size(8) 
                self.ax1.xaxis.get_offset_text().set_size(8) 
            if n == 1:
                self.ax2 = self.fig.add_subplot( 1, num, 2)
                self.ax2.tick_params(axis='both', which='major', labelsize=8)
                self.ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax2.yaxis.get_offset_text().set_size(8) 
            if n == 2:
                self.ax3 = self.fig.add_subplot( 1, num, 3) 
                self.ax3.tick_params(axis='both', which='major', labelsize=8)
                self.ax3.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax3.yaxis.get_offset_text().set_size(8) 
            if n == 3:
                self.ax4 = self.fig.add_subplot( 1, num, 4) 
                self.ax4.tick_params(axis='both', which='major', labelsize=8)
                self.ax4.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax4.yaxis.get_offset_text().set_size(8) 
    
    def reAxH2(self, num, shx=True, shy=True):
        try:
            for ax in fig.axes:
                self.fig.delaxes(ax)
        except:
            pass
        try:
            self.fig.clear()
        except:
            pass
        
        for n in range(num):
            if n == 0:
                self.ax1 = self.fig.add_subplot( 2, num, 1)
                self.ax1.tick_params(axis='both', which='major', labelsize=8)
                self.ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax1.yaxis.get_offset_text().set_size(8) 
                self.ax21 = self.fig.add_subplot( 2, num, num+1)
                self.ax21.tick_params(axis='both', which='major', labelsize=8)
                self.ax21.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax21.yaxis.get_offset_text().set_size(8) 
            if n == 1:
                self.ax2 = self.fig.add_subplot( 2, num, 2, sharex=self.ax1, sharey=self.ax1)
                self.ax2.tick_params(axis='both', which='major', labelsize=8)
                self.ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax2.yaxis.get_offset_text().set_size(8) 
                self.ax22 = self.fig.add_subplot( 2, num, num+2, sharex=self.ax21, sharey=self.ax21)
                self.ax22.tick_params(axis='both', which='major', labelsize=8)
                self.ax22.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax22.yaxis.get_offset_text().set_size(8) 
            if n == 2:
                self.ax3 = self.fig.add_subplot( 2, num, 3, sharex=self.ax1, sharey=self.ax1)
                self.ax3.tick_params(axis='both', which='major', labelsize=8)
                self.ax3.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax3.yaxis.get_offset_text().set_size(8) 
                self.ax23 = self.fig.add_subplot( 2, num, num+3, sharex=self.ax21, sharey=self.ax21)
                self.ax23.tick_params(axis='both', which='major', labelsize=8)
                self.ax23.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax23.yaxis.get_offset_text().set_size(8) 
            if n == 3:
                self.ax4 = self.fig.add_subplot( 2, num, 4, sharex=self.ax1, sharey=self.ax1 )
                self.ax4.tick_params(axis='both', which='major', labelsize=8)
                self.ax4.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax4.yaxis.get_offset_text().set_size(8) 
                self.ax24 = self.fig.add_subplot( 2, num, num+4, sharex=self.ax21, sharey=self.ax21 )
                self.ax24.tick_params(axis='both', which='major', labelsize=8)
                self.ax24.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax24.yaxis.get_offset_text().set_size(8) 
            if n == 4:
                self.ax5 = self.fig.add_subplot( 2, num, 5, sharex=self.ax1, sharey=self.ax1 )
                self.ax5.tick_params(axis='both', which='major', labelsize=8)
                self.ax5.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax5.yaxis.get_offset_text().set_size(8) 
                self.ax25 = self.fig.add_subplot( 2, num, num+5, sharex=self.ax21, sharey=self.ax21 )
                self.ax25.tick_params(axis='both', which='major', labelsize=8)
                self.ax25.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax25.yaxis.get_offset_text().set_size(8) 
            if n == 5:
                self.ax6 = self.fig.add_subplot( 2, num, 6, sharex=self.ax1, sharey=self.ax1 )
                self.ax6.tick_params(axis='both', which='major', labelsize=8)
                self.ax6.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax6.yaxis.get_offset_text().set_size(8) 
                self.ax26 = self.fig.add_subplot( 2, num, num+6, sharex=self.ax21, sharey=self.ax21 )
                self.ax26.tick_params(axis='both', which='major', labelsize=8)
                self.ax26.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax26.yaxis.get_offset_text().set_size(8) 
            if n == 6:
                self.ax7 = self.fig.add_subplot( 2, num, 7, sharex=self.ax1, sharey=self.ax1 )
                self.ax7.tick_params(axis='both', which='major', labelsize=8)
                self.ax7.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax7.yaxis.get_offset_text().set_size(8) 
                self.ax27 = self.fig.add_subplot( 2, num, num+7, sharex=self.ax21, sharey=self.ax21 )
                self.ax27.tick_params(axis='both', which='major', labelsize=8)
                self.ax27.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax27.yaxis.get_offset_text().set_size(8) 
            if n == 7:
                self.ax8 = self.fig.add_subplot( 2, num, 8, sharex=self.ax1, sharey=self.ax1 )
                self.ax8.tick_params(axis='both', which='major', labelsize=8)
                self.ax8.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax8.yaxis.get_offset_text().set_size(8) 
                self.ax28 = self.fig.add_subplot( 2, num, num+8, sharex=self.ax21, sharey=self.ax21 )
                self.ax28.tick_params(axis='both', which='major', labelsize=8)
                self.ax28.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
                self.ax28.yaxis.get_offset_text().set_size(8) 

    def reAx2(self, shx=True, shy=True):

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
            self.delaxes(self.ax3) #.clear()
        except:
            pass
        
        try:
            self.ax2.clear() 
            self.delaxes(self.ax2) #.clear()
        except:
            pass

        #self.fig.patch.set_facecolor('red')
        self.ax1 = self.fig.add_subplot(211)
        if shx and shy:
            self.ax2 = self.fig.add_subplot(212, sharex=self.ax1, sharey=self.ax1)
        elif shx == True:
            self.ax2 = self.fig.add_subplot(212, sharex=self.ax1) 
        elif shy == True:
            self.ax2 = self.fig.add_subplot(212, sharey=self.ax1) 
        else:
            self.ax2 = self.fig.add_subplot(212)

        self.ax1.tick_params(axis='both', which='major', labelsize=8)
        self.ax2.tick_params(axis='both', which='major', labelsize=8)

        self.ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
        self.ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  

        self.ax1.yaxis.get_offset_text().set_size(8) 
        self.ax2.yaxis.get_offset_text().set_size(8) 

    def reAx3(self, shx=True, shy=True):

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
        
        try:    
            self.ax3.clear() 
            self.delaxes(self.ax3) #.clear()
        except:
            pass

        self.ax1 = self.fig.add_subplot(211)
        if shx and shy:
            self.ax2 = self.fig.add_subplot(212, sharex=self.ax1, sharey=self.ax1)
        elif shx:
            self.ax2 = self.fig.add_subplot(212, sharex=self.ax1) 
        elif shy:
            self.ax2 = self.fig.add_subplot(212, sharey=self.ax1) 
        else:
            self.ax2 = self.fig.add_subplot(212) 

        self.ax3 = self.ax1.twinx()

        #self.ax1.set_facecolor('red')
        #self.ax2.set_facecolor('red')
        #self.ax3.set_facecolor('red')
        #self.fig.set_facecolor('red')
        #self.fig.set_edgecolor('red')
        #self.ax1.set_axis_bgcolor('green')

        self.ax1.tick_params(axis='both', which='major', labelsize=8)
        self.ax2.tick_params(axis='both', which='major', labelsize=8)
        self.ax3.tick_params(axis='both', which='major', labelsize=8)

        self.ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
        self.ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  
        self.ax3.ticklabel_format(style='sci', scilimits=(0,0), axis='y')  

        self.ax1.yaxis.get_offset_text().set_size(8) 
        self.ax2.yaxis.get_offset_text().set_size(8)    
        self.ax3.yaxis.get_offset_text().set_size(8)    
 
    def reAx4(self):

        try:
            self.fig.clear()
        except:
            pass

        # two main axes
        self.ax1 = self.fig.add_axes([0.15, 0.55,   0.625, 0.3672])
        self.ax2 = self.fig.add_axes([0.15, 0.135,  0.625, 0.3672])
        
        # for colourbars
        self.cax1 = self.fig.add_axes([0.8, 0.55,   0.025, 0.3672])
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


    def compute_initial_figure(self):
        
        t = np.arange(0,.3,1e-4)
        x = np.cos(t*2000.*np.pi*2)*np.exp(-t/.07)
        x2 = np.exp(-t/.07)
        dp = self.ax1.plot(t, x, 'r',label='test function')
        dp2 = self.ax2.plot(t, x2, 'r',label='test function2')
        self.ax1.set_xlabel("Time [s]", fontsize=8)
        self.ax1.set_ylabel("Signal [nV]", fontsize=8)

        self.ax1.tick_params(axis='both', which='major', labelsize=8)
        self.ax1.tick_params(axis='both', which='minor', labelsize=6)

        self.ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y') 
        self.ax1.legend(prop={'size':6})

