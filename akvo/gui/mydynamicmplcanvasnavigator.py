from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

import sys
from PyQt5 import QtCore, QtGui

#from mydynamicmplcanvas import MyMplCanvas

class MyMplCanvasN(FigureCanvas):
    
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=3, height=.2, dpi=100):
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)

        self.setParent(parent)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

    def clicked(self):
        print ("Clicked")

class MyDynamicMplCanvasNavigator(MyMplCanvasN):
    
    def __init__(self, *args, **kwargs):
        
        MyMplCanvasN.__init__(self, *args, **kwargs)

    def setCanvas(self, canvas):
        NavigationToolbar(canvas, self)
