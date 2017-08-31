import sys
import numpy as np
import matplotlib.pyplot as plt
import plotyaml

__author__ = "M. Andy Kass"
__version__ = "$Revision 0.1"
__date__ = "$Date: 2017-05-25"


class USGSPlots:

    def getData(self,fname,chan):
            
        self.channels = []
        for part in chan.split(','):
            if ':' in part:
                a,b = part.split(':')
                a,b = int(a), int(b)
                self.channels.extend(range(a,b))
            else:
                a = int(part)
                self.channels.append(a)
        #print self.channels

        self.data = []
        for c in self.channels:
            nm = "Chan. " + str(c)
            self.data.append(plotyaml.loadAkvoData(fname,nm))


        return 0

    def plotSingleDecay(self,pm,chan=1):
       

        return 42


    def plotAllDecays(self,chan=1):

        return 42

    def plotAverageFID(self,chan=1):

        return 42

    def plotSpectrum(self,chan=1):

        return 42
 
