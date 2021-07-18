
import os, sys
import numpy as np
from ruamel import yaml

import pyLemma.LemmaCore as lc 
import pyLemma.Merlin as mrln 
import pyLemma.FDEM1D as em1d 

import numpy as np

#import matplotlib.pyplot as plt 
#import seaborn as sns
#sns.set(style="ticks")
#import cmocean 
#from SEGPlot import *
#from matplotlib.ticker import FormatStrFormatter
#import matplotlib.ticker as plticker


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


def main():

    if len(sys.argv) < 3:
        print ("usage  python calcAkvoKernel.py   AkvoDataset.yaml  Coil1.yaml  " )
        exit()

    AKVO = loadAkvoData(sys.argv[1])

    B_inc = AKVO.META["B_0"]["inc"]  
    B_dec = AKVO.META["B_0"]["dec"]  
    B0 = AKVO.META["B_0"]["intensity"]  

    gamma = 2.67518e8
    fT = AKVO.transFreq
    #B0 = (fL*2.*np.pi) /gamma * 1e9
 
    Coil1 = em1d.PolygonalWireAntenna.DeSerialize( sys.argv[2] )
    Coil1.SetNumberOfFrequencies(1)
    Coil1.SetFrequency(0, fT) 
    Coil1.SetCurrent(1.)

    lmod = em1d.LayeredEarthEM() 
    lmod.SetNumberOfLayers(4)
    lmod.SetLayerThickness([15.49, 28.18])
    lmod.SetLayerConductivity([0.0, 1./16.91, 1./24.06, 1./33.23])

    lmod.SetMagneticFieldIncDecMag( B_inc, B_dec, B0, lc.NANOTESLA )
    
    exit()

    Kern = mrln.KernelV0()
    Kern.PushCoil( "Coil 1", Coil1 )
    Kern.SetLayeredEarthEM( lmod );
    Kern.SetIntegrationSize( (200,200,200) )
    Kern.SetIntegrationOrigin( (0,0,0) )
    Kern.SetTolerance( 1e-9 )
    Kern.SetMinLevel( 3 )
    Kern.SetHankelTransformType( lc.FHTKEY201 )
    Kern.AlignWithAkvoDataset( sys.argv[1] )

    thick = np.geomspace(.5, 10,num=40)
    iface = np.cumsum(thick)
    Kern.SetDepthLayerInterfaces(iface)
    #Kern.SetDepthLayerInterfaces(np.geomspace(1, 110, num=40))
    #Kern.SetDepthLayerInterfaces(np.linspace(1, 110, num=50))
    #Kern.SetDepthLayerInterfaces(np.geomspace(1, 110, num=40))
 
    # autAkvoDataNode = YAML::LoadFile(argv[4]);
    # Kern->AlignWithAkvoDataset( AkvoDataNode );

    #Kern.SetPulseDuration(0.040)
    #Kern.SetPulseCurrent(  [1.6108818092452406, 1.7549935078885168, 1.7666319459646016, 1.9270787752430283,
    #    1.9455431806179229, 2.111931346726564, 2.1466747256211747, 2.3218217392379588,
    #    2.358359967649008, 2.5495654202189058, 2.5957289164577992, 2.8168532605800802,
    #    2.85505242699187, 3.1599429539069774, 3.2263673040205068, 3.6334182368296544,
    #    3.827985200119751, 4.265671313014058, 4.582237014873297, 5.116839616183394,
    #    5.515173073160611, 6.143620383280934, 6.647972282096122, 7.392577402979211,
    #    8.020737177449933, 8.904435233295793, 9.701975105606063, 10.74508217792577,
    #    11.743887525923592, 12.995985956061467, 14.23723766879807, 15.733870137824457,
    #    17.290155933625808, 19.07016662950366, 21.013341340455703, 23.134181634845618,
    #    25.570925414182238, 28.100862178905476, 31.13848909847073, 34.16791099558486,
    #    37.95775984680512, 41.589619321873165, 46.327607251605286, 50.667786337299205,
    #    56.60102493062895, 61.81174065797068, 69.23049946198458, 75.47409803238031,
    #    84.71658869065816, 92.1855007134236, 103.77129947551164, 112.84577430578537,
    #    127.55127257092909, 138.70199812969176, 157.7443764728878, 171.39653462998626]
    #)

    Kern.CalculateK0( ["Coil 1"], ["Coil 1"], False )

    yml = open('akvoK3-' + str(Kern.GetTolerance()) + '.yaml', 'w')
    print(Kern, file=yml)

    K0 = Kern.GetKernel()
    
    #plt.matshow(np.abs(K0))
    #plt.show()

if __name__ == "__main__":
    main()
