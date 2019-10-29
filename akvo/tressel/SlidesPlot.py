#################################################################################
# GJI final pub specs                                                           #
import matplotlib                                                               #
from matplotlib import rc                                                           #

#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{timet,amsmath,amssymb}"]  #
#rc('font',**{'family':'sans-serif','serif':['timet']})                                   #
#rc('font',**{'size':11})                                                             #
#rc('text', usetex=True)                                                             #

# converts pc that GJI is defined in to inches                                      # 
# In GEOPHYSICS \textwidth = 42pc                                               #
#        \columnwidth = 20pc                                                    #
#        one column widthe figures are 20 picas                                 #
#        one and one third column figures are 26 picas                          #
def pc2in(pc):                                                                  #
    return pc*12/72.27                                                          #
#################################################################################
import numpy as np
light_grey = np.array([float(248)/float(255)]*3)

def fixLeg(legend):
    rect = legend.get_frame()
    #rect.set_color('None')
    rect.set_facecolor(light_grey)
    rect.set_linewidth(0.0)
    rect.set_alpha(0.5)

def deSpine(ax1):
    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax1.spines[spine].set_visible(False)
    #ax1.xaxis.set_ticks_position('none')
    #ax1.yaxis.set_ticks_position('none')
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
