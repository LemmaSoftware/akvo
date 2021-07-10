
import os, sys
import numpy as np
from ruamel import yaml

import pyLemma.LemmaCore as lc 
import pyLemma.Merlin as mrln 
import pyLemma.FDEM1D as em1d 

import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(style="ticks")

from ruamel import yaml

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

    if len(sys.argv) < 2:
        print ("usage  python calcAkvoKernel.py   AkvoDataset.yaml  Coil1.yaml kparams.yaml  SaveString.yaml " )
        print ("usage  akvoKO   AkvoDataset.yaml   kparams.yaml  SaveString.yaml " )
        exit()

    AKVO = loadAkvoData(sys.argv[1])

    B_inc = AKVO.META["B_0"]["inc"]  
    B_dec = AKVO.META["B_0"]["dec"]  
    B0    = AKVO.META["B_0"]["intensity"]  

    fT = AKVO.transFreq
    #gamma = 2.67518e8
    #B0 = (fL*2.*np.pi) /gamma * 1e9
    
    # read in kernel params
    kparams = loadAkvoData( sys.argv[2] )
 
    Kern = mrln.KernelV0()

    TX = []
    for tx in kparams['txCoils']:
        Coil1 = em1d.PolygonalWireAntenna.DeSerialize( tx )
        Coil1.SetNumberOfFrequencies(1)
        Coil1.SetFrequency(0, fT) 
        Coil1.SetCurrent(1.)
        Kern.PushCoil( tx.split('.yml')[0], Coil1 )
        TX.append( tx.split('.yml')[0] )
    
    RX = []
    for rx in kparams['rxCoils']:
        if rx not in kparams['txCoils']:
            print("new recv")
            Coil1 = em1d.PolygonalWireAntenna.DeSerialize( rx )
            Coil1.SetNumberOfFrequencies(1)
            Coil1.SetFrequency(0, fT) 
            Coil1.SetCurrent(1.)
            Kern.PushCoil( rx.split('.yml')[0], Coil1 )
        else:
            print("reuse tx coil")
        RX.append( rx.split('.yml')[0] )
    

    ## TODO 
    # pass this in...
    lmod = em1d.LayeredEarthEM() 

    nlay = len(kparams["sigs"])
    sigs = np.array(kparams["sigs"])
    tops = np.array(kparams["tops"])
    bots = np.array(kparams["bots"])
    
    if ( (len(tops)-1) != len(bots)):
        print("Layer mismatch")
        exit()

    thicks = bots - tops[0:-1]
    
    lmod.SetNumberOfLayers(nlay + 1)
    lmod.SetLayerThickness(thicks)
    lmod.SetLayerConductivity( np.concatenate( ( [0.0], sigs ) ))

    #lmod.SetNumberOfLayers(4)
    #lmod.SetLayerThickness([15.49, 28.18])
    #lmod.SetLayerConductivity([0.0, 1./16.91, 1./24.06, 1./33.23])


    lmod.SetMagneticFieldIncDecMag( B_inc, B_dec, B0, lc.NANOTESLA )


    Kern.SetLayeredEarthEM( lmod );
    Kern.SetIntegrationSize( (kparams["size_n"], kparams["size_e"], kparams["size_d"]) )
    Kern.SetIntegrationOrigin( (kparams["origin_n"], kparams["origin_e"], kparams["origin_d"]) )
    Kern.SetTolerance( 1e-9*kparams["branchTol"] )
    Kern.SetMinLevel( kparams["minLevel"] )
    Kern.SetMaxLevel( kparams["maxLevel"] )
    Kern.SetHankelTransformType( lc.FHTKEY201 )
    Kern.AlignWithAkvoDataset( sys.argv[1] )

    if str(kparams["Lspacing"]).strip() == "Geometric":
        thick = np.geomspace(kparams["thick1"], kparams["thickN"], num=kparams["nLay"])
    elif str(kparams["Lspacing"]) == "Log":
        thick = np.logspace(kparams["thick1"], kparams["thickN"], num=kparams["nLay"])
    elif str(kparams["Lspacing"]) == "Linear":
        thick = np.linspace(kparams["thick1"], kparams["thickN"], num=kparams["nLay"])
    else:
        print("DOOOM!, in calcAkvoKernel layer spacing was not <Geometric>, <Log>, or <Linear>")
        print( str(kparams["Lspacing"]) )
        exit()

    iface = np.cumsum(thick)
    Kern.SetDepthLayerInterfaces(iface)
    #Kern.SetDepthLayerInterfaces(np.geomspace(1, 110, num=40))
    #Kern.SetDepthLayerInterfaces(np.linspace(1, 110, num=50))
    #Kern.SetDepthLayerInterfaces(np.geomspace(1, 110, num=40))
 
    # autAkvoDataNode = YAML::LoadFile(argv[4]);
    # Kern->AlignWithAkvoDataset( AkvoDataNode );

    #Kern.CalculateK0( ["Coil 1"], ["Coil 1"], False )
    Kern.CalculateK0( TX, RX, False )

    #yml = open( 'test' + str(Kern.GetTolerance()) + '.yaml', 'w')
    yml = open( sys.argv[3], 'w' )
    print(Kern, file=yml)

    # 
    K0 = Kern.GetKernel()
    plt.matshow(np.abs(K0))
    plt.show()

if __name__ == "__main__":
    main()
