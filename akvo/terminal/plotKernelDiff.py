import matplotlib.pyplot as plt
import sys,os
from pylab import meshgrid
from matplotlib.colors import LightSource
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import yaml
from  lemma_yaml import *

import cmocean 

def catLayers(K0):
    K = np.zeros( (len(K0.keys()), len(K0["layer-0"].data)) , dtype=complex )
    for lay in range(len(K0.keys())):
        #print(K0["layer-"+str(lay)].data) #    print (lay)
        K[lay] = K0["layer-"+str(lay)].data #    print (lay)
    return K

if __name__ == "__main__":

    with open(sys.argv[1]) as f:
        # use safe_load instead load
        K0 = yaml.load(f, Loader=yaml.Loader)
    
    with open(sys.argv[2]) as f2:
        # use safe_load instead load
        K02 = yaml.load(f2, Loader=yaml.Loader)

    K = 1e9*catLayers(K0.K0)
    K2 = 1e9*catLayers(K02.K0)
    
    q = np.array(K0.PulseI.data)* (float)(K0.Taup)

    #plt.pcolor(K0.Interfaces.data, K0.PulseI.data, np.abs(K))
    plt.pcolor(q, K0.Interfaces.data, np.abs(K), cmap=cmocean.cm.gray_r)
    #plt.contourf(q, K0.Interfaces.data[0:-1], np.abs(K), cmap=cmocean.cm.tempo)
    #plt.pcolor(q, K0.Interfaces.data, np.abs(K), cmap=cmocean.cm.tempo)
    
    plt.pcolor(q, K0.Interfaces.data, np.abs(K - K2), cmap=cmocean.cm.gray_r)
    plt.colorbar()

    ax1 = plt.gca()
    ax1.set_ylim( ax1.get_ylim()[1], ax1.get_ylim()[0]  )
    #ax1.set_xscale('log')
    #ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.set_xticks([ax1.get_xlim()[0], 1, ax1.get_xlim()[1],])
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    #print(yaml.dump(K0.K0))
    #print( K0.K0["layer-0"].data )
    #print( type(  np.array(K0.K0["layer-0"].data) ) )
    #plt.plot( np.real( K0.K0["layer-0"].data ) )
    #plt.plot( K0.K0["layer-0"].data )
    plt.show()
    #print(yaml.dump(K0))
