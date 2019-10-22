import yaml
    
def Complex_ctor(loader, node):
    re = eval(node.value[0][1].value)
    im = eval(node.value[1][1].value)
    return re + 1j*im 

yaml.add_constructor(r'Complex', Complex_ctor)

class MatrixXr(yaml.YAMLObject):
    yaml_tag = u'MatrixXr'
    def __init__(self, rows, cols, data):
        self.rows = rows
        self.cols = cols
        self.data = np.zeros((rows,cols))
    def __repr__(self):
        return "%s(rows=%r, cols=%r, data=%r)" % (self.__class__.__name__, self.rows, self.cols, self.data) 

class VectorXr(yaml.YAMLObject):
    yaml_tag = r'VectorXr'
    def __init__(self, array):
        self.size = np.shape(array)[0]
        self.data = array.tolist()
    def __repr__(self):
        # Converts to numpy array on import 
        return "np.array(%r)" % (self.data)

class VectorXcr(yaml.YAMLObject):
    yaml_tag = r'VectorXcr'
    def __init__(self, array):
        self.size = np.shape(array)[0]
        self.datar = array.tolist()
    def __repr__(self):
        # Converts to numpy array on import 
        #return "np.array(%r)" % (self.data)
        return "np.array(%r)" % (3)

class Vector3r(yaml.YAMLObject):
    yaml_tag = r'Vector3r'
    def __init__(self, array):
        self.size = 3 #np.shape(array)[0]
        self.data = array.tolist()
    def __repr__(self):
        # Converts to numpy array on import 
        return "np.array(%r)" % (self.data)

class Vector3Xcr(yaml.YAMLObject):
    yaml_tag = r'Vector3Xcr'
    def __init__(self, array):
        self.size = 3 #np.shape(array)[0]
        self.data = array.tolist()
    def __repr__(self):
        # Converts to numpy array on import 
        return "np.array(%r)" % (self.data)

class Vector3Xr(yaml.YAMLObject):
    yaml_tag = r'Vector3Xr'
    def __init__(self, array):
        self.size = 3 #np.shape(array)[0]
        self.data = array.tolist()
    def __repr__(self):
        # Converts to numpy array on import 
        return "np.array(%r)" % (self.data)

#class KernelV0( ):
    #yaml_tag = r'KernelV0'
#    def __init__(self):
#        self.name = "hello"

#def KernelV0_constructor(loader, node):
    #...     value = loader.construct_scalar(node)
    #...     a, b = map(int, value.split('d'))
#    return KernelV0( )


# class KervnelV0(yaml.YAMLObject):
#     yaml_loader = yaml.Loader
#     yaml_dumper = yaml.Dumper
# 
#     yaml_tag = u'!KernelV0'
#     #yaml_flow_style = ...
# 
#     def __init__(self):
#         self.val = 7
# 
#     @classmethod
#     def from_yaml(cls, loader, node):
#         # ...
#         data = 0
#         return data
# 
#     @classmethod
#     def to_yaml(cls, dumper, data):
#         # ...
#         return node

class KervnelV0(yaml.YAMLObject):
    yaml_tag = u'KernelV0'
    def __init__(self, val):
        self.val = val

class LayeredEarthEM(yaml.YAMLObject):
    yaml_tag = u'LayeredEarthEM'
    def __init__(self, val):
        self.val = val

class PolygonalWireAntenna(yaml.YAMLObject):
    yaml_tag = u'PolygonalWireAntenna'
    def __init__(self, val):
        self.val = val

class AkvoData(yaml.YAMLObject):
    yaml_tag = u'AkvoData'
    def __init__(self, obj): #nPulseMoments, pulseLength):
    #def __init__(self, rows, cols, data):
        #self.nPulseMoments = nPulseMoments
        #self.pulseLength = pulseLength 
        #for key in obj.keys:
        #    self[key] = obj.key
        pass

