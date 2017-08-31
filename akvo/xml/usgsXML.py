# XML metadata writer suitable for USGS Sciencebase Data Release
# M. Andy Kass
# 2017-02-17


class Akvo(sNMRdata):

    def __init__(self):
        return

    def readProcessed(self,fname):
        # Load in the saved pickle saved from Akvo

        return 42

    def readInverted(self,fname):
        return 42

    def readGenericInfo(self,gfname='basicinfo,yaml',sfname='sNMRinfo.yaml'):
        super().readGenericInfo(gfname)
        super().readsNMRinfo(sfname)

    def importProcessed(self,fname):
        # Load in the exported YAML data from Akvo

        return 42

class VCsurfProcessed(sNMRdata):

    def __init__(self):
        return

class VCDartdata(bNMRdata):

    def __init__(self):
        print('VCDartdata')
        return


class VCJavelindata(bNMRdata):

    def __init__(self):
        return




# ----------Should never call these directly-------------------

class data():
    
    def __init__(self):
        return

    def readGenericInfo(self,fname):
        # Read in the generic information common to all surveys.

        return 42

class NMRdata(data):

    def __init__(self):
        return

class bNMRdata(NMRdata):

    def __init__(self):
        return

class sNMRdata(NMRdata):

    def __init__(self):
        return

    def readsNMRinfo(self,fname):
        # Read in generic info common to all sNMR surveys

        return 42


