from ruamel import yaml
import os, sys
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
import cmocean 
from SEGPlot import *
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as plticker

def slicedict(d, s):
    return {k:v for k,v in d.items() if k.startswith(s)}

# Converts Lemma/Merlin/Akvo serialized Eigen arrays into numpy ones for use by Python 
class VectorXr(yaml.YAMLObject):
    """
    Converts Lemma/Merlin/Akvo serialized Eigen arrays into numpy ones for use by Python 
    """
    yaml_tag = u'VectorXr'
    def __init__(self, array):
        self.size = np.shape(array)[0]
        self.data = array.tolist()
    def __repr__(self):
        # Converts to numpy array on import 
        return "np.array(%r)" % (self.data)

class AkvoData(yaml.YAMLObject):
    """
    Reads an Akvo serialized dataset into a standard python dictionary 
    """
    yaml_tag = u'AkvoData'
    def __init__(self, array):
        pass
        #self.size = np.shape(array)[0]
        #self.Imp = array.tolist()
    def __repr__(self):
        # Converts to a dictionary with Eigen vectors represented as Numpy arrays 
        return self

def loadAkvoData(fnamein):
    """ Loads data from an Akvo YAML file. The 0.02 is hard coded as the pulse length. This needs to be 
        corrected in future kernel calculations. The current was reported but not the pulse length. 
    """
    fname = (os.path.splitext(fnamein)[0])
    with open(fnamein, 'r') as stream:
        try:
            AKVO = (yaml.load(stream, Loader=yaml.Loader))
        except yaml.YAMLError as exc:
            print(exc)
    return AKVO 

def plotQt( akvo ):
    plt.style.use('ggplot')
    #plt.style.use('seaborn-white')
    for pulse in akvo.Gated:
        if pulse[0:5] == "Pulse":
            #print(akvo.GATED[pulse].keys())
            nq = akvo.Pulses[pulse]["current"].size
            for chan in slicedict(akvo.Gated[pulse], "Chan."):
                # accumulate pulse moments
                CA = np.zeros( (nq, len( akvo.Gated[pulse]["abscissa"].data )) )
                RE = np.zeros( (nq, len( akvo.Gated[pulse]["abscissa"].data )) )
                IM = np.zeros( (nq, len( akvo.Gated[pulse]["abscissa"].data )) )
                for q in range(nq):
                    #plt.plot(  akvo.Gated[pulse]["abscissa"].data,  akvo.Gated[pulse][chan]["Q-" + str(q)+" CA"].data ) 
                    CA[q] = akvo.Gated[pulse][chan]["Q-" + str(q)+" CA"].data
                    RE[q] = akvo.Gated[pulse][chan]["Q-" + str(q)+" RE"].data
                    IM[q] = akvo.Gated[pulse][chan]["Q-" + str(q)+" IM"].data
                    #X[q] = akvo.Gated[pulse][chan]["Q-" + str(q)+" RE"].data
            Windows = akvo.Gated[pulse]["windows"].data
            Q = np.array(akvo.Pulses[pulse]["current"].data)
            print("pulse length ", akvo.pulseLength[0])
            Q *= akvo.pulseLength[0]
       
            fig = plt.figure( figsize=( pc2in(20), pc2in(26) ) ) 
            ax1 = fig.add_axes([.25,.05,.6,.9])
            im = ax1.pcolormesh(Windows,Q,CA, cmap=cmocean.cm.curl_r, vmin=-np.max(np.abs(CA)), vmax=(np.max(np.abs(CA))))
            cb = plt.colorbar( im, orientation='horizontal', pad=.175,  )
            cb.set_label("FID (nV)", fontsize=10)
            cb.ax.tick_params(labelsize=10) 
            ax1.set_yscale('log')
            ax1.set_ylabel("Q (A$\cdot$s)", fontsize=10)
            ax1.set_xscale('log')
            ax1.set_xlabel("time (s)", fontsize=10)

            #loc = plticker.MultipleLocator(25) #base=50.0) # this locator puts ticks at regular intervals
            #loc = plticker.MaxNLocator(5, steps=[1,2]) #base=50.0) # this locator puts ticks at regular intervals
            #ax1.xaxis.set_major_locator(loc)

            #ax1.xaxis.set_minor_locator(plticker.NullLocator())
            ax1.xaxis.set_major_formatter(FormatStrFormatter('%2.0f'))
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%2.1f'))

            #plt.figure()
            #sns.kdeplot(Windows, Q, CA) #, kind="hex", color="#4CB391")
            #sns.heatmap(CA, annot=False, center=0)

    #plt.matshow(RE)
    #plt.matshow(IM)
    plt.savefig("data.pgf")
    plt.savefig("data.png", dpi=300)
    #plt.show()

if __name__ == "__main__":
    akvo = loadAkvoData( sys.argv[1] ) #, "Chan. 1")
    plotQt(akvo)
