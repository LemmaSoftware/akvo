import yaml
import os, sys
import numpy as np

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

def loadAkvoData(fnamein, chan):
    """ Loads data from an Akvo YAML file. The 0.02 is hard coded as the pulse length. This needs to be 
        corrected in future kernel calculations. The current was reported but not the pulse length. 
    """
    fname = (os.path.splitext(fnamein)[0])
    with open(fnamein, 'r') as stream:
        try:
            AKVO = (yaml.load(stream))
        except yaml.YAMLError as exc:
            print(exc)
    return AKVO 

def plotQt( akvo ):
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    for pulse in akvo.Gated:
        if pulse[0:5] == "Pulse":
            #print(akvo.GATED[pulse].keys())
            nq = akvo.Pulses[pulse]["current"].size
            for chan in slicedict(akvo.Gated[pulse], "Chan."):
                # accumulate pulse moments
                X = np.zeros( (nq, len( akvo.Gated[pulse]["abscissa"].data )) )
                for q in range(nq):
                    plt.plot(  akvo.Gated[pulse]["abscissa"].data,  akvo.Gated[pulse][chan]["Q-" + str(q)+" CA"].data ) 
                    X[q] = akvo.Gated[pulse][chan]["Q-" + str(q)+" CA"].data
    plt.matshow(X)

    plt.show()
if __name__ == "__main__":
    akvo = loadAkvoData( sys.argv[1] , "Chan. 1")
    plotQt(akvo)
