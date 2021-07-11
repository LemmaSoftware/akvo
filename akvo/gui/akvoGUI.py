#/usr/bin/env python 
import sys

import matplotlib
matplotlib.use("QT5Agg")
from PyQt5 import QtCore, QtGui, QtWidgets #, uic
import numpy as np
import time
import os
from copy import deepcopy
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT #as NavigationToolbar
import datetime, time
import pkg_resources  # part of setuptools
from collections import OrderedDict
from ruamel import yaml

from akvo.gui.main_ui import Ui_MainWindow
from akvo.gui.addCircularLoop_ui import Ui_circularLoopAdd
from akvo.gui.addFigure8Loop_ui import Ui_figure8LoopAdd
from akvo.gui.addPolygonalLoop_ui import Ui_polygonalLoopAdd
from akvo.gui.redirect_ui import Ui_callScript
from akvo.gui.callScript import callScript

from akvo.tressel import mrsurvey 

from pyLemma import LemmaCore  
from pyLemma import FDEM1D 
from pyLemma import Merlin

VERSION = pkg_resources.require("Akvo")[0].version
GAMMAH = 42.577478518 * 1e-3  # Hz nT

# Writes out numpy arrays into Eigen vectors as serialized by Lemma
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


def setup_yaml():
    """ https://stackoverflow.com/a/8661021 """
    represent_dict_order = lambda self, data:  self.represent_mapping('tag:yaml.org,2002:map', data.items())
    yaml.add_representer(OrderedDict, represent_dict_order)    
setup_yaml()

class AkvoYamlNode(yaml.YAMLObject):
    yaml_tag = u'AkvoData'
    def __init__(self):
        self.Akvo_VERSION = VERSION
        self.Import = OrderedDict() # {}
        self.Processing = [] # OrderedDict() 
        self.Stacking = OrderedDict()
        self.META = OrderedDict() 
    def __repr__(self):
        return "%s(name=%r, Akvo_VERSION=%r, Import=%r, Processing=%r, self.Stacking=%r, self.META=%r)" % (
            self.__class__.__name__, self.Akvo_VERSION, self.Import, self.Processing, self.Stacking, self.META )
    
try:    
    import thread 
except ImportError:
    import _thread as thread  #Py3K compatibility 

class MyPopup(QtWidgets.QWidget):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.initUI()

    def initUI(self):
        lblName = QtWidgets.QLabel(self.name, self)

class ApplicationWindow(QtWidgets.QMainWindow):
    
    def __init__(self):
        
        super().__init__() 
        #QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)      

        # alternative to calling pyuic 
        #self.ui = uic.loadUi('main.ui', self) 

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
    
        self.RAWDataProc = None 
        self.YamlNode = AkvoYamlNode()           
 
        # initialise some stuff
        self.ui.lcdNumberTauPulse2.setEnabled(0)
        self.ui.lcdNumberTauPulse1.setEnabled(0)
        self.ui.lcdNumberNuTx.setEnabled(0)
        self.ui.lcdNumberTuneuF.setEnabled(0)
        self.ui.lcdNumberSampFreq.setEnabled(0)
        self.ui.lcdNumberTauDelay.setEnabled(0)
        self.ui.lcdNumberNQ.setEnabled(0)

        self.logText = []      

        #######################
        ##################### #
        ## Make connections # #
        ##################### #
        #######################

        ##############
        # Menu items #
        ##############
        self.ui.actionOpen_GMR.triggered.connect(self.openGMRRAWDataset)
        self.ui.actionSave_Preprocessed_Dataset.triggered.connect(self.SavePreprocess)
        self.ui.actionExport_Preprocessed_Dataset.triggered.connect(self.ExportPreprocess)
        self.ui.actionExport_Preprocessed_Dataset.setEnabled(False)
        self.ui.actionOpen_Preprocessed_Dataset.triggered.connect(self.OpenPreprocess)
        self.ui.actionAboutAkvo.triggered.connect(self.about)

        ###########
        # Buttons #
        ###########
        self.ui.loadDataPushButton.pressed.connect(self.loadRAW) 
        self.ui.sumDataGO.pressed.connect( self.sumDataChans )
        self.ui.bandPassGO.pressed.connect( self.bandPassFilter )
        self.ui.filterDesignPushButton.pressed.connect( self.designFilter )
        self.ui.fdDesignPushButton.pressed.connect( self.designFDFilter )
        self.ui.downSampleGO.pressed.connect( self.downsample )
        self.ui.windowFilterGO.pressed.connect( self.windowFilter )
        self.ui.adaptGO.pressed.connect( self.adaptFilter )
        self.ui.adaptFDGO.pressed.connect( self.adaptFilterFD )
        self.ui.qdGO.pressed.connect( self.quadDet )
        self.ui.gateIntegrateGO.pressed.connect( self.gateIntegrate )
        self.ui.calcQGO.pressed.connect( self.calcQ )
        self.ui.FDSmartStackGO.pressed.connect( self.FDSmartStack )
        self.ui.harmonicGO.pressed.connect( self.harmonicModel )       
        
        self.ui.K0Data.pressed.connect( self.K0DataSelect )       
        
        self.ui.invDataButton.pressed.connect( self.invDataSelect )       
        self.ui.invKernelButton.pressed.connect( self.invKernelSelect )       

        self.ui.f0K1Spin.valueChanged.connect( self.LCDHarmonics )
        self.ui.f0KNSpin.valueChanged.connect( self.LCDHarmonics )
        self.ui.f0KsSpin.valueChanged.connect( self.LCDHarmonics )
        self.ui.f0Spin.valueChanged.connect( self.LCDHarmonics )

        self.ui.NHarmonicsFreqsSpin.valueChanged.connect( self.LCDHarmonics2 )
        
        self.ui.f1K1Spin.valueChanged.connect( self.LCDHarmonics2 )
        self.ui.f1KNSpin.valueChanged.connect( self.LCDHarmonics2 )
        self.ui.f1KsSpin.valueChanged.connect( self.LCDHarmonics2 )
        self.ui.f1Spin.valueChanged.connect( self.LCDHarmonics2 )
 
        self.ui.plotQD.setEnabled(False) 
        self.ui.plotQD.pressed.connect( self.plotQD )
        
        self.ui.plotGI.setEnabled(False) 
        self.ui.plotGI.pressed.connect( self.plotGI )

        # balance the Larmor frequency info and Tx off resonance info
        self.ui.intensitySpinBox.valueChanged.connect( self.adjustLarmor )
        self.ui.txv.valueChanged.connect( self.adjustB0 )
        self.ui.larmorv.valueChanged.connect( self.adjustB02 )

        # Kernel         
        self.ui.calcK0.pressed.connect( self.calcK0 )       

        # Inversion
        self.ui.invertButton.pressed.connect( self.QTInv )

        # META 
        self.ui.locEdit.editingFinished.connect( self.logSite )
        self.ui.UTMzone.currentIndexChanged.connect( self.logSite )
        self.ui.latBand.currentIndexChanged.connect( self.logSite )
        self.ui.ellipsoid.currentIndexChanged.connect( self.logSite )
        self.ui.incSpinBox.valueChanged.connect( self.logSite )
        self.ui.decSpinBox.valueChanged.connect( self.logSite )
        self.ui.intensitySpinBox.valueChanged.connect( self.logSite )
        self.ui.tempSpinBox.valueChanged.connect( self.logSite )
        self.ui.timeEdit.timeChanged.connect( self.logSite )
        self.ui.dateEdit.dateChanged.connect( self.logSite )
        # this may call the yaml stuff too often... 
        self.ui.txtComments.textChanged.connect( self.logSite )
        self.ui.plotLoops.pressed.connect( self.plotLoops2 )       
        self.ui.removeLoopButton.pressed.connect( self.removeLoop )       
 
        # Loops
        self.ui.addLoopButton.pressed.connect( self.loopAdd )
        self.loops = {}          
 
        # hide header info box 
        #self.ui.headerFileBox.setVisible(False) 
        self.ui.headerFileBox.clicked.connect( self.headerBoxShrink ) 
        self.ui.headerBox2.setVisible(False) 

        # Clean up the tab widget 
        self.ui.actionPreprocessing.triggered.connect(self.addPreProc)
        self.ui.actionModelling.triggered.connect(self.addModelling)
        self.ui.actionInversion.triggered.connect(self.addInversion)

        # tabs 
        #self.ui.ProcTabs.tabCloseRequested.connect( self.closeTabs )
        #self.ui.ProcTabs.tabBar().setTabButton(7, QtWidgets.QTabBar.RightSide,None) 
        self.ui.ProcTabs.removeTab(4)    
        self.ui.ProcTabs.removeTab(4)    
        self.ui.ProcTabs.removeTab(4)    
        self.ui.ProcTabs.removeTab(4)    
        #self.ui.LoadTab.close(  ) 

        # Add progressbar to statusbar
        self.ui.barProgress =  QtWidgets.QProgressBar()
        self.ui.statusbar.addPermanentWidget(self.ui.barProgress, 0);
        self.ui.barProgress.setMaximumSize(100, 16777215);
        self.ui.barProgress.hide();        

        self.ui.mplwidget_navigator.setCanvas(self.ui.mplwidget)
        #self.ui.mplwidget_navigator_2.setCanvas(self.ui.mplwidget)

        self.ui.txRxTable.setColumnCount(4)
        self.ui.txRxTable.setRowCount(0)
        self.ui.txRxTable.setHorizontalHeaderLabels( ["Label", "Geom.","Turns","Tx/Rx"] )
        
        ##########################################################################
        # layer Table 
        self.ui.layerTableWidget.setRowCount(80)       
        self.ui.layerTableWidget.setColumnCount(3)      
        self.ui.layerTableWidget.setHorizontalHeaderLabels( [r"top [m]", r"bottom [m]", "ρ [Ωm]" ] )

        # do we want this
        self.ui.layerTableWidget.setDragDropOverwriteMode(False)
        self.ui.layerTableWidget.setDragEnabled(True)
        self.ui.layerTableWidget.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        
        pCell0 = QtWidgets.QTableWidgetItem()
        pCell0.setFlags(QtCore.Qt.NoItemFlags) # not selectable 
        pCell0.setBackground( QtGui.QColor("lightgrey").lighter(110) )
        pCell0.setForeground( QtGui.QColor("black") )
        pCell0.setText(str("0"))
        self.ui.layerTableWidget.setItem(0, 0, pCell0)
        
        pCell1 = QtWidgets.QTableWidgetItem()
        #pCell1.setFlags(QtCore.Qt.NoItemFlags) # not selectable 
        pCell1.setBackground( QtGui.QColor("lightblue") ) #.lighter(110) )
        pCell1.setForeground( QtGui.QColor("black") )
        self.ui.layerTableWidget.setItem(0, 1, pCell1)
        
        pCell2 = QtWidgets.QTableWidgetItem()
        #pCell1.setFlags(QtCore.Qt.NoItemFlags) # not selectable 
        pCell2.setBackground( QtGui.QColor("white") ) #.lighter(110) )
        pCell2.setForeground( QtGui.QColor("black") )
        self.ui.layerTableWidget.setItem(0, 2, pCell2)


        for ir in range(1, self.ui.layerTableWidget.rowCount() ):
            for ic in range(0, self.ui.layerTableWidget.columnCount() ):
                pCell = QtWidgets.QTableWidgetItem()
                #pCell.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                pCell.setFlags(QtCore.Qt.NoItemFlags) # not selectable 
                pCell.setBackground( QtGui.QColor("lightgrey").lighter(110) )
                pCell.setForeground( QtGui.QColor("black"))
                self.ui.layerTableWidget.setItem(ir, ic, pCell)
        self.ui.layerTableWidget.cellChanged.connect(self.sigmaCellChanged) 

    def LCDHarmonics(self):
        self.ui.lcdH1F.setEnabled(True)
        self.ui.lcdH1F.display( self.ui.f0Spin.value() * self.ui.f0K1Spin.value() )
        self.ui.lcdHNF.setEnabled(True)
        self.ui.lcdHNF.display( self.ui.f0Spin.value() * self.ui.f0KNSpin.value() )
        self.ui.lcdf0NK.setEnabled(True)
        self.ui.lcdf0NK.display( (self.ui.f0KNSpin.value()+1-self.ui.f0K1Spin.value()) * self.ui.f0KsSpin.value() )
    
    def LCDHarmonics2(self):
        if self.ui.NHarmonicsFreqsSpin.value() == 2:
            self.ui.lcdH1F2.setEnabled(True)
            self.ui.lcdH1F2.display( self.ui.f1Spin.value() * self.ui.f1K1Spin.value() )
            self.ui.lcdHNF2.setEnabled(True)
            self.ui.lcdHNF2.display( self.ui.f1Spin.value() * self.ui.f1KNSpin.value() )
            self.ui.lcdf0NK2.setEnabled(True)
            self.ui.lcdf0NK2.display( (self.ui.f1KNSpin.value()+1-self.ui.f1K1Spin.value()) * self.ui.f1KsSpin.value() )
        else:
            self.ui.lcdH1F2.setEnabled(False)
            self.ui.lcdHNF2.setEnabled(False)
            self.ui.lcdf0NK2.setEnabled(False)

    def adjustLarmor(self):
        """ Triggers when the B0 intensity spin box is cycled
        """
        self.ui.larmorv.setValue( self.ui.intensitySpinBox.value() * GAMMAH )
        self.ui.txv.setValue( self.RAWDataProc.transFreq - self.ui.larmorv.value() )

    def adjustB0(self):
        """ Triggers when tx frequency offset is cycled
        """
        self.ui.intensitySpinBox.setValue( (self.RAWDataProc.transFreq - self.ui.txv.value()) / GAMMAH )
    
    def adjustB02(self):
        """ Triggers when Larmor frequency spin box is cycled o
        """
        self.ui.intensitySpinBox.setValue( (self.ui.larmorv.value()) / GAMMAH )

    def closeTabs(self):
        #self.ui.ProcTabs.removeTab(idx)    
        self.ui.ProcTabs.clear( )    
    
    def addPreProc(self):
        if self.ui.actionPreprocessing.isChecked(): 
            self.ui.actionModelling.setChecked(False) 
            self.ui.actionInversion.setChecked(False) 
            self.ui.ProcTabs.clear( )    
            self.ui.ProcTabs.insertTab( 0, self.ui.LoadTab, "Load" )    
            self.ui.ProcTabs.insertTab( 1, self.ui.NCTab, "Noise removal" )    
            self.ui.ProcTabs.insertTab( 2, self.ui.QCTab, "QC" )    
            self.ui.ProcTabs.insertTab( 3, self.ui.METATab, "META" )    
            self.ui.ProcTabs.insertTab( 4, self.ui.LogTab, "Log" )    
        else:
            self.ui.ProcTabs.removeTab(0)    
            self.ui.ProcTabs.removeTab(0)    
            self.ui.ProcTabs.removeTab(0)    
            self.ui.ProcTabs.removeTab(0)    
    
    def addModelling(self):
        if self.ui.actionModelling.isChecked(): 
            self.ui.actionPreprocessing.setChecked(False) 
            self.ui.actionInversion.setChecked(False) 
            self.ui.ProcTabs.clear( )    
            self.ui.ProcTabs.insertTab( 0, self.ui.KernTab, "Kernel" )    
            self.ui.ProcTabs.insertTab( 1, self.ui.ModelTab, "Modelling" )    
            #self.ui.ProcTabs.insertTab( 2, self.ui.LogTab, "Log" )    
        else:
            self.ui.ProcTabs.removeTab(0)    
            self.ui.ProcTabs.removeTab(0)    
    
    def addInversion(self, idx):
        if self.ui.actionInversion.isChecked(): 
            self.ui.actionPreprocessing.setChecked(False) 
            self.ui.actionModelling.setChecked(False) 
            self.ui.ProcTabs.clear( )    
            self.ui.ProcTabs.insertTab( 0, self.ui.InvertTab, "QT Inversion" )    
            self.ui.ProcTabs.insertTab( 1, self.ui.AppraiseTab, "Appraisal" )    
            #self.ui.ProcTabs.insertTab( 2, self.ui.LogTab, "Log" )    
        else:
            self.ui.ProcTabs.removeTab(0)    
            self.ui.ProcTabs.removeTab(0)    
    
    def invDataSelect(self):
        try:
            with open('.akvo.last.path') as f: 
                fpath = f.readline()  
                pass
        except IOError as e:
            fpath = '.'
        
        self.akvoDataFile = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Datafile File', fpath, r"Akvo datafiles (*.akvoProcData *.yaml)")[0]
        self.ui.dataText.clear()   
        self.ui.dataText.append( self.akvoDataFile )   
    
    def K0DataSelect(self):
        try:
            with open('.akvo.last.path') as f: 
                fpath = f.readline()  
                pass
        except IOError as e:
            fpath = '.'
        
        self.K0akvoDataFile = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Datafile File', fpath, r"Akvo datafiles (*.akvoProcData *.yaml)")[0]

        # populate combo box with loops 
        if os.path.isfile( self.K0akvoDataFile ): 
        
            self.ui.K0DataText.clear()   
            self.ui.K0DataText.append( self.K0akvoDataFile )  

            with open(self.K0akvoDataFile) as f:
                parse = yaml.load( f, Loader=yaml.Loader )
        
            self.ui.txListWidget.clear()
            self.ui.rxListWidget.clear()
            for loop in parse.META["Loops"]:
                print(loop)
                self.ui.txListWidget.addItem( parse.META["Loops"][loop] )
                self.ui.rxListWidget.addItem( parse.META["Loops"][loop] )
           
            # TODO, why are these necessary 
            self.ui.txListWidget.setCurrentRow(0)
            self.ui.rxListWidget.setCurrentRow(0)
        # else do nothing
         
    def invKernelSelect(self):
        try:
            with open('.akvo.last.path') as f: 
                fpath = f.readline()  
                pass
        except IOError as e:
            fpath = '.'
        
        self.K0file = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Kernel File', fpath, r"Akvo kernels (*.akvoK0)")[0]
        self.ui.kernelText.clear()   
        self.ui.kernelText.append(self.K0file)   
    
    def QTInv(self):
    
        print("Big RED INVERT BUTTON")
        
        try:
            with open('.akvo.last.path') as f: 
                fpath = f.readline()  
                pass
        except IOError as e:
            fpath = '.'
        
        #K0file = self.ui.kernelText.text()
        #akvoDataFile = self.ui.dataText.text()   

        T2lo = self.ui.T2low.value()
        T2hi = self.ui.T2hi.value()
        NT2 = self.ui.NT2.value()
        dataChan = self.ui.invChan.currentText()

        t2Obj = self.ui.T2Objective.currentText()
        depthObj = self.ui.depthObjective.currentText()
        alpha_0 = self.ui.initialAlpha.value()

        invDict = dict()
        invDict["data"] = dict()
        invDict["data"] = dict() 
        invDict["data"][self.akvoDataFile] = dict()
        invDict["data"][self.akvoDataFile]["channels"] = [dataChan,]
        invDict["K0"] = [self.K0file,]
        invDict["T2Bins"] = dict()
        invDict["T2Bins"]["low"] = T2lo
        invDict["T2Bins"]["high"] = T2hi
        invDict["T2Bins"]["number"] = NT2       
        invDict["NonLinearRefinement"] = self.ui.NLButton.isChecked()
        invDict["CalcDOI"] = self.ui.DOIButton.isChecked()
 
        node = yaml.YAML()
        kpo = open( "invert.yml", 'w' )
        node.dump(invDict, kpo)
 
        callBox = callScript(  ) #QtWidgets.QDialog() 
        
        callBox.ui = Ui_callScript()
        callBox.ui.setupUi( callBox )
        callBox.setupQTInv( "invert.yml" )
        
        callBox.exec_()
        callBox.show()
            

    def calcK0(self):
        
        try:
            with open('.akvo.last.path') as f: 
                fpath = f.readline()  
                pass
        except IOError as e:
            fpath = '.'

        #self.K0akvoDataFile = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Datafile File', fpath, r"Akvo datafiles (*.yaml)")[0]
        #akvoData = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Datafile File', fpath, r"Akvo datafiles (*.yaml)")[0]

        txCoilList = self.ui.txListWidget.selectedItems() #currentItem().text() 
        txCoils = []
        for txCoil in txCoilList:
            print("txCoil", txCoil.text())
            txCoils.append(txCoil.text())
        
        rxCoilList = self.ui.txListWidget.selectedItems() #currentItem().text() 
        rxCoils = []
        for rxCoil in rxCoilList:
            print("rxCoil", rxCoil.text())
            rxCoils.append(rxCoil.text())


        saveStr = QtWidgets.QFileDialog.getSaveFileName(self, "Save kernel as", fpath, r"Merlin KernelV0 (*.akvoK0)")[0] 

        intDict = dict()
        intDict["origin_n"] = self.ui.originN.value()
        intDict["origin_e"] = self.ui.originE.value()
        intDict["origin_d"] = self.ui.originD.value()
        intDict["size_n"] = self.ui.sizeN.value()
        intDict["size_e"] = self.ui.sizeE.value()
        intDict["size_d"] = self.ui.sizeD.value()
        intDict["nLay"] = self.ui.NLayers.value()
        intDict["thick1"] = self.ui.thick1.value()
        intDict["thickN"] = self.ui.thickN.value()
        intDict["Lspacing"] = self.ui.layerSpacing.currentText()
        intDict["minLevel"] = self.ui.minLevel.value()
        intDict["maxLevel"] = self.ui.maxLevel.value()
        intDict["branchTol"] = self.ui.branchTol.value()

        intDict["txCoils"] = txCoils
        intDict["rxCoils"] = rxCoils

        # conductivity model... 
        #tops = self.ui.layerTableWidget.col(0)
        #print("Tops", tops)
        tops = []
        itop = 0
        while self.ui.layerTableWidget.item(itop, 0).text(): 
            tops.append( float(self.ui.layerTableWidget.item(itop,0).text()) ) 
            itop += 1

        bots = []
        ibot = 0
        while self.ui.layerTableWidget.item(ibot, 1).text(): 
            bots.append( float(self.ui.layerTableWidget.item(ibot, 1).text()) ) 
            ibot += 1
        
        sigs = []
        isig = 0
        while self.ui.layerTableWidget.item(isig, 2).text(): 
            sigs.append( 1./float(self.ui.layerTableWidget.item(isig, 2).text()) ) 
            isig += 1

        intDict["tops"] = tops
        intDict["bots"] = bots
        intDict["sigs"] = sigs

        node = yaml.YAML()
        kpo = open( "kparams.yml", 'w' )
        node.dump(intDict, kpo)

        callBox = callScript(  ) #QtWidgets.QDialog() 
        
        callBox.ui = Ui_callScript()
        callBox.ui.setupUi( callBox )
        callBox.setupCB( self.K0akvoDataFile, "kparams.yml", saveStr  )
        
        callBox.exec_()
        callBox.show()
    
    def loopAdd(self):

        #print(self.ui.loopLabel.text())
        #print(self.ui.loopGeom.currentText())
        #print(self.ui.loopType.currentText())
        #print( "label len", len(self.ui.loopLabel.text()) ) 
        if len(self.ui.loopLabel.text().strip()) == 0: 
            Error = QtWidgets.QMessageBox()
            Error.setWindowTitle("Error!")
            Error.setText("Loop label cannot be blank or repeated")
            Error.setDetailedText("Each loop label must be unique and comprise at least one character. Leading and trailing whitespace will be trimmed.")
            Error.exec_()
        else:
            
            ### Circular loop
            if self.ui.loopGeom.currentText() == "Circular":
                dialog = QtWidgets.QDialog()
                dialog.ui = Ui_circularLoopAdd()
                dialog.ui.setupUi(dialog)
                dialog.exec_()
                dialog.show()

                if dialog.result():
                    cn = dialog.ui.centreNorth.value()
                    ce = dialog.ui.centreEast.value()
                    ht = dialog.ui.loopHeight.value()
                    rad = dialog.ui.loopRadius.value()
                    turns = dialog.ui.loopTurns.value()
                    ns = dialog.ui.segments.value()
                    cwise = dialog.ui.cwiseBox.currentIndex()
                    #print("cwise", cwise)
                    #dip = dialog.ui.dip.value()
                    #azimuth = dialog.ui.az.value()
                    
                    self.loops[self.ui.loopLabel.text()] = FDEM1D.PolygonalWireAntenna()
                    self.loops[self.ui.loopLabel.text()].SetNumberOfPoints( dialog.ui.segments.value() + 1 )
                    self.loops[self.ui.loopLabel.text()].SetNumberOfTurns( dialog.ui.loopTurns.value() )

                    points = np.linspace(0, 2*np.pi, dialog.ui.segments.value()+1)
                    for iseg, ipt in enumerate(points):
                        if cwise == 0:
                            self.loops[self.ui.loopLabel.text()].SetPoint(iseg, ( cn+rad*np.sin(ipt), ce+rad*np.cos(ipt), ht) )
                        else:
                            self.loops[self.ui.loopLabel.text()].SetPoint(iseg, ( cn-rad*np.sin(ipt), ce+rad*np.cos(ipt), ht) )
                    self.loops[self.ui.loopLabel.text()].SetNumberOfFrequencies(1)
                    self.loops[self.ui.loopLabel.text()].SetCurrent(1.)
            
            if self.ui.loopGeom.currentText() == "figure-8":
                dialog = QtWidgets.QDialog()
                dialog.ui = Ui_figure8LoopAdd()
                dialog.ui.setupUi(dialog)
                dialog.exec_()
                dialog.show()
                
                if dialog.result():
                    cn1 = dialog.ui.centreNorth1.value()
                    ce1 = dialog.ui.centreEast1.value()
                    cn2 = dialog.ui.centreNorth2.value()
                    ce2 = dialog.ui.centreEast2.value()
                    ht = dialog.ui.loopHeight.value()
                    rad = dialog.ui.loopRadius.value()
                    turns = dialog.ui.loopTurns.value()
                    ns = dialog.ui.segments.value()
                    cwise = dialog.ui.cwiseBox.currentIndex()

                    self.loops[self.ui.loopLabel.text()] = FDEM1D.PolygonalWireAntenna()
                    self.loops[self.ui.loopLabel.text()].SetNumberOfPoints( 2*dialog.ui.segments.value() + 1 )
                    self.loops[self.ui.loopLabel.text()].SetNumberOfTurns( dialog.ui.loopTurns.value() )

                    # first loop
                    points = np.linspace(0, 2*np.pi, dialog.ui.segments.value())
                    ptsL = []
                    for iseg, ipt in enumerate(points):
                        ptsL.append( np.array( [cn1+rad*np.sin(ipt), ce1+rad*np.cos(ipt)]  ))
                    lenP = len(points)
                    
                    # search for closest point, ugly and not efficient, but it's not critical here 
                    closest = 1e8
                    iclosest = -1
                    for iseg, ipt in enumerate(points):
                        p2 = np.array([cn2-rad*np.sin(ipt), ce2-rad*np.cos(ipt)])
                        for p1 in ptsL:
                            dist = np.linalg.norm(p1-p2)
                            if dist < closest:
                                closest = dist
                                iclosest = iseg

                    points = np.concatenate([points[iclosest::],points[0:iclosest]])

                    # Fill first loop
                    point1 = False
                    for iseg, ipt in enumerate(points):
                        if cwise == 0:
                            self.loops[self.ui.loopLabel.text()].SetPoint(iseg, ( cn1+rad*np.sin(ipt), ce1+rad*np.cos(ipt), ht) )
                            pointlast = ( cn1+rad*np.sin(ipt), ce1+rad*np.cos(ipt), ht) 
                            if not point1:
                                point1 = ( cn1+rad*np.sin(ipt), ce1+rad*np.cos(ipt), ht) 
                        else:
                            self.loops[self.ui.loopLabel.text()].SetPoint(iseg, ( cn1-rad*np.sin(ipt), ce1+rad*np.cos(ipt), ht) )
                            if not point1:
                                point1 = ( cn1-rad*np.sin(ipt), ce1+rad*np.cos(ipt), ht) 
                            pointlast = ( cn1-rad*np.sin(ipt), ce1+rad*np.cos(ipt), ht) 

                    lenP = len(points)

                    # reorder points again to find nearest point in second loop
                    closest = 99999
                    iclosest = -1 
                    for iseg, ipt in enumerate(points):
                        if cwise == 0:
                            p2 = np.array([cn2-rad*np.sin(ipt), ce2+rad*np.cos(ipt)])
                        else:
                            p2 = np.array([cn2+rad*np.sin(ipt), ce2+rad*np.cos(ipt)])
                        for p1 in ptsL:
                            dist = np.linalg.norm(np.array(pointlast[0:2])-p2)
                            if dist < closest:
                                closest = dist
                                iclosest = iseg
                    points = np.concatenate([points[iclosest::],points[0:iclosest]])

                    # fill second loop
                    for iseg, ipt in enumerate(points):
                        if cwise == 0:
                            self.loops[self.ui.loopLabel.text()].SetPoint(lenP+iseg, ( cn2-rad*np.sin(ipt), ce2+rad*np.cos(ipt), ht) )
                        else:
                            self.loops[self.ui.loopLabel.text()].SetPoint(lenP+iseg, ( cn2+rad*np.sin(ipt), ce2+rad*np.cos(ipt), ht) )

                    # close loop        
                    self.loops[self.ui.loopLabel.text()].SetPoint(lenP+iseg+1, point1) 


                    self.loops[self.ui.loopLabel.text()].SetNumberOfFrequencies(1)
                    self.loops[self.ui.loopLabel.text()].SetCurrent(1.)
 
            
            if self.ui.loopGeom.currentText() == "polygon":
                
                dialog = QtWidgets.QDialog()
                dialog.ui = Ui_polygonalLoopAdd()
                dialog.ui.setupUi(dialog)

                ##########################################################################
                # Loop Table 
                dialog.ui.loopTableWidget.setRowCount(80)       
                dialog.ui.loopTableWidget.setColumnCount(3)  
                #dialog.ui.loopTableWidget.horizontalHeader().setSectionResizeMode(0, QtGui.Qt.QHeaderView.Stretch)  
                dialog.ui.loopTableWidget.setHorizontalHeaderLabels( \
                    ["Northing [m]","Easting [m]","Height [m]"])

 
                for ir in range(0, dialog.ui.loopTableWidget.rowCount() ):
                    for ic in range(0, 3): #dialog.ui.loopTableWidget.columnCount() ):
                        pCell = QtWidgets.QTableWidgetItem()
                        #pCell.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                        #pCell.setFlags(QtCore.Qt.NoItemFlags) # not selectable 
                        #pCell.setBackground( QtGui.QColor("lightgrey").lighter(110) )
                        dialog.ui.loopTableWidget.setItem(ir, ic, pCell)

                #dialog.ui.loopTableWidget.cellChanged.connect(self.loopCellChanged)
                #dialog.ui.loopTableWidget.itemClicked.connect(self.loopCellClicked) 
                
                #self.ui.loopTableWidget.cellPressed.connect(self.loopCellChanged) 
                #self.ui.loopTableWidget.cellPressed.connect(self.loopCellClicked) 

                dialog.ui.loopTableWidget.setDragDropOverwriteMode(False)
                dialog.ui.loopTableWidget.setDragEnabled(False)
                #self.ui.loopTableWidget.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

                dialog.ui.loopTableWidget.resizeColumnsToContents()       
                dialog.exec_()
                dialog.show()
                
                if dialog.result():
                    self.loops[self.ui.loopLabel.text()] = FDEM1D.PolygonalWireAntenna()
                    self.loops[self.ui.loopLabel.text()].SetNumberOfTurns( dialog.ui.loopTurns.value() )
               
                    npts = 0 
                    for ir in range(0, dialog.ui.loopTableWidget.rowCount() ):
                        if len(dialog.ui.loopTableWidget.item(ir, 0).text()) == 0:
                            break
                        npts += 1 
                    self.loops[self.ui.loopLabel.text()].SetNumberOfPoints( npts )
                    for ir in range( 0, npts ):
                        self.loops[self.ui.loopLabel.text()].SetPoint(ir, eval(dialog.ui.loopTableWidget.item(ir, 0).text()),  \
                                                                          eval(dialog.ui.loopTableWidget.item(ir, 1).text()),  \
                                                                          eval(dialog.ui.loopTableWidget.item(ir, 2).text())   \
                                                                     ) 
                    self.loops[self.ui.loopLabel.text()].SetNumberOfFrequencies(1)
                    self.loops[self.ui.loopLabel.text()].SetCurrent(1.)


            # general across all types   
            if dialog.result():

                yml = open( self.ui.loopLabel.text() + ".pwa", 'w' )
                print( self.loops[self.ui.loopLabel.text()], file=yml)

                # update the table 
                self.ui.txRxTable.setRowCount( len(self.loops.keys()) )
                    
                pCell = QtWidgets.QTableWidgetItem()
                pCell.setText( self.ui.loopLabel.text() ) 
                pCell.setFlags( QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled )
                self.ui.txRxTable.setItem( len(self.loops.keys())-1, 0, pCell)
                    
                gCell = QtWidgets.QTableWidgetItem()
                gCell.setText( self.ui.loopGeom.currentText() ) 
                gCell.setFlags( QtCore.Qt.ItemIsEnabled )
                self.ui.txRxTable.setItem( len(self.loops.keys())-1, 1, gCell)
                    
                tCell = QtWidgets.QTableWidgetItem()
                tCell.setText( str(dialog.ui.loopTurns.value()) ) 
                tCell.setFlags( QtCore.Qt.ItemIsEnabled )
                self.ui.txRxTable.setItem( len(self.loops.keys())-1, 2, tCell)
                    
                txCell = QtWidgets.QTableWidgetItem()
                txCell.setText( str(self.ui.loopType.currentText()) ) 
                txCell.setFlags( QtCore.Qt.ItemIsEnabled )
                self.ui.txRxTable.setItem( len(self.loops.keys())-1, 3, txCell)

    def headerBoxShrink(self):
        #self.ui.headerFileBox.setVisible(False)
 
        if self.ui.headerFileBox.isChecked( ): 
            #self.ui.headerFileBox.setMinimumSize(460,250) 
            self.ui.headerBox2.setVisible(True) 
        else:
            #self.ui.headerFileBox.setMinimumSize(460,50) 
            self.ui.headerBox2.setVisible(False) 

    def sigmaCellChanged(self):
        self.ui.layerTableWidget.cellChanged.disconnect(self.sigmaCellChanged) 
        # TODO consider building the model whenever this is called. Would be nice to be able to 
        # do that. Would require instead dist of T2 I guess.
        jj = self.ui.layerTableWidget.currentColumn()
        ii = self.ui.layerTableWidget.currentRow()
        val = "class 'NoneType'>"
        try:
            val = eval (str( self.ui.layerTableWidget.item(ii, jj).text() ))
        except:
            #if jj != 0:
            #    Error = QtWidgets.QMessageBox()
            #    Error.setWindowTitle("Error!")
            #    Error.setText("Non-numeric value encountered")
            self.ui.layerTableWidget.cellChanged.connect(self.sigmaCellChanged) 
            return
        if jj == 1:
            #item.setFlags(QtCore.Qt.ItemIsEnabled)
            pCell = self.ui.layerTableWidget.item(ii, jj)
            pCell.setBackground( QtGui.QColor("white"))
            
            pCell = self.ui.layerTableWidget.item(ii+1, jj-1)
            if str(type(pCell)) == "<class 'NoneType'>":  
                pCell = QtWidgets.QTableWidgetItem()
                pCell.setFlags(QtCore.Qt.ItemIsEnabled)
                self.ui.layerTableWidget.setItem(ii+1, jj-1, pCell)
            if ii == 0:
                pCell.setText(str(val))
                #pCell3 = self.ui.layerTableWidget.item(ii+1, jj)
                #print ("setting", ii, jj, type(pCell3))
                #print ( "setting", ii, jj, type(pCell3))
                #pCell3.setFlags( QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled )
                #pCell3.setFlags( QtCore.Qt.ItemIsEditable  )
            elif ii > 0:
                val2 = eval (str( self.ui.layerTableWidget.item(ii-1, jj).text() ))
                #print ("val2", val2, val, type(val))
                #if str(type(pCell)) == "<class 'NoneType'>":  
                if type(val) == str or val > val2:
                    pCell.setText(str(val))
                else:
                    Error = QtWidgets.QMessageBox()
                    Error.setWindowTitle("Error!")
                    Error.setText("Non-increasing layer detected")
                    Error.setDetailedText("Each layer interface must be below the one above it.")
                    Error.exec_()
                    #err_msg = "Quadrature detection has already been done!"
                    #reply =QtWidgets.QMessageBox.critical(self, 'Error', 
                    #    err_msg) 

                    pCell2 = self.ui.layerTableWidget.item(ii, jj)
                    pCell2.setText(str(""))
                    self.ui.layerTableWidget.cellChanged.connect(self.sigmaCellChanged) 
                    return

            # enable next layer
            pCell4 = self.ui.layerTableWidget.item(ii+1, jj)
            pCell4.setBackground( QtGui.QColor("lightblue") ) #.lighter(110))
            pCell4.setForeground( QtGui.QColor("black"))
            pCell4.setFlags( QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled )
            
            pCell5 = self.ui.layerTableWidget.item(ii+1, jj+1)
            pCell5.setBackground( QtGui.QColor("white"))
            pCell5.setForeground( QtGui.QColor("black"))
            pCell5.setFlags( QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled )
            
        if ii == 0 and jj == 0:
            pCell = self.ui.layerTableWidget.item(0, 1)
            pCell.setBackground(QtGui.QColor("lightblue")) #.lighter(110) )
            pCell.setForeground( QtGui.QColor("black"))
            pCell.setFlags( QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled )

        self.ui.layerTableWidget.cellChanged.connect(self.sigmaCellChanged) 

    def plotLoops2(self):
        self.ui.mplwidget.reAxH(1)

        for loop in self.loops:
            POINTS = self.loops[loop].GetPoints().T
            self.ui.mplwidget.ax1.plot( POINTS[:,1], POINTS[:,0], label=loop )
       
        self.ui.mplwidget.ax1.spines['right'].set_visible(False)
        self.ui.mplwidget.ax1.spines['top'].set_visible(False) 
        self.ui.mplwidget.ax1.set_xlabel("easting (m)")
        self.ui.mplwidget.ax1.set_ylabel("northing (m)")
        self.ui.mplwidget.ax1.legend()
        self.ui.mplwidget.ax1.set_aspect('equal') #, adjustable='box')
        self.ui.mplwidget.draw()
    
    def removeLoop(self):
        del self.loops[ self.ui.txRxTable.item( self.ui.txRxTable.currentRow(), 0).text() ]
        self.ui.txRxTable.removeRow(self.ui.txRxTable.currentRow()) 

    def about(self):
        # TODO proper popup with info
        #self.w = MyPopup("""About Akvo \n
        #    Akvo is an open source project developed primarily by Trevor Irons. 
        #""")
        #self.w.setGeometry(100, 100, 400, 200)
        #self.w.show()
        #print("ABOUT")

        # Just a splash screen for now
        logo = pkg_resources.resource_filename(__name__, 'akvo-about.png')
        pixmap = QtGui.QPixmap(logo)
        self.splash = QtWidgets.QSplashScreen(pixmap, QtCore.Qt.WindowStaysOnTopHint)
        self.splash.show()

    def connectGMRDataProcessor(self):
        
        self.RAWDataProc = mrsurvey.GMRDataProcessor()
        self.RAWDataProc.progressTrigger.connect(self.updateProgressBar)
        self.RAWDataProc.enableDSPTrigger.connect(self.enableDSP)
        self.RAWDataProc.doneTrigger.connect(self.doneStatus)
        self.RAWDataProc.updateProcTrigger.connect(self.updateProc)

    def openGMRRAWDataset(self):
        """ Opens a GMR header file
        """
        try:
            with open('.gmr.last.path') as f: 
                fpath = f.readline()  
                pass
        except IOError as e:
            fpath = '.'

        self.headerstr = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', fpath)[0] # arg2 = File Type 'All Files (*)'
        self.ui.headerFileTextBrowser.clear()
        self.ui.headerFileTextBrowser.append(self.headerstr)
        
        if len(self.headerstr) == 0:
            return 
        
        # clear the processing log
        self.ui.logTextBrowser.clear()
        self.logText = []   #MAK 20170126

        path,filen=os.path.split(str(self.headerstr))

        f = open('.gmr.last.path', 'w')
        f.write( str(self.headerstr) ) # prompt last file      

        self.connectGMRDataProcessor()
        self.RAWDataProc.readHeaderFile(str(self.headerstr))

        # If we got this far, enable all the widgets
        self.ui.lcdNumberTauPulse1.setEnabled(True)
        self.ui.lcdNumberNuTx.setEnabled(True)
        self.ui.lcdNumberTuneuF.setEnabled(True)
        self.ui.lcdNumberSampFreq.setEnabled(True)
        self.ui.lcdNumberNQ.setEnabled(True)

        self.ui.headerFileBox.setEnabled(True)
        self.ui.headerFileBox.setChecked( True )
        self.ui.headerBox2.setVisible(True) 
        self.ui.inputRAWParametersBox.setEnabled(True)
        self.ui.loadDataPushButton.setEnabled(True)
         
        # make plots as you import the dataset
        self.ui.plotImportCheckBox.setEnabled(True)
        self.ui.plotImportCheckBox.setChecked(True)
         
        # Update info from the header into the GUI
        self.ui.pulseTypeTextBrowser.clear()
        self.ui.pulseTypeTextBrowser.append(self.RAWDataProc.pulseType)
        self.ui.lcdNumberNuTx.display(self.RAWDataProc.transFreq)
        self.ui.lcdNumberTauPulse1.display(1e3*self.RAWDataProc.pulseLength[0])
        self.ui.lcdNumberTuneuF.display(self.RAWDataProc.TuneCapacitance)
        self.ui.lcdNumberSampFreq.display(self.RAWDataProc.samp)
        self.ui.lcdNumberNQ.display(self.RAWDataProc.nPulseMoments)
        self.ui.DeadTimeSpinBox.setValue(1e3*self.RAWDataProc.deadTime)
        self.ui.CentralVSpinBox.setValue( self.RAWDataProc.transFreq )

        # set the B0 field according to Tx as an initial guess
        self.ui.intensitySpinBox.setValue( self.RAWDataProc.transFreq/GAMMAH )           
 
        if self.RAWDataProc.pulseType != "FID":
            self.ui.lcdNumberTauPulse2.setEnabled(1)
            self.ui.lcdNumberTauPulse2.display(1e3*self.RAWDataProc.pulseLength[1])
            self.ui.lcdNumberTauDelay.setEnabled(1)
            self.ui.lcdNumberTauDelay.display(1e3*self.RAWDataProc.interpulseDelay)

        self.ui.FIDProcComboBox.clear() 
        if self.RAWDataProc.pulseType == "4PhaseT1" or self.RAWDataProc.pulseType == "T1":
            self.ui.FIDProcComboBox.insertItem(0, "Pulse 1") 
            self.ui.FIDProcComboBox.insertItem(1, "Pulse 2") 
            self.ui.FIDProcComboBox.insertItem(2, "Both")    
            self.ui.FIDProcComboBox.setCurrentIndex (1)
        elif self.RAWDataProc.pulseType == "FID":
            self.ui.FIDProcComboBox.insertItem(0, "Pulse 1") 
            self.ui.FIDProcComboBox.setCurrentIndex (0)
    
    def ExportPreprocess(self):
        """ This method exports to YAML 
        """
        try:
            with open('.akvo.last.yaml.path') as f: 
                fpath = f.readline()  
                pass
        except IOError as e:
            fpath = '.'
        
        fdir = os.path.dirname(fpath)
        # Pickle the preprocessed data dictionary 
        SaveStr = QtWidgets.QFileDialog.getSaveFileName(self, "Save as", fdir, r"Processed data (*.akvoProcData)")[0]
        
        spath,filen=os.path.split(str(SaveStr))
        f = open('.akvo.last.yaml.path', 'w')
        f.write( str(spath) ) # prompt last file      

        INFO = {}
        INFO["headerstr"] = str(self.headerstr)
        INFO["pulseType"] = self.RAWDataProc.pulseType
        INFO["transFreq"] = self.RAWDataProc.transFreq.tolist()
        INFO["pulseLength"] = self.RAWDataProc.pulseLength.tolist()
        INFO["TuneCapacitance"] = self.RAWDataProc.TuneCapacitance.tolist()
        #INFO["samp"] = self.RAWDataProc.samp
        INFO["nPulseMoments"] = self.RAWDataProc.nPulseMoments
        #INFO["deadTime"] = self.RAWDataProc.deadTime
        INFO["processed"] = "Akvo v"  + VERSION + ", on "  + time.strftime("%d/%m/%Y")
        # Pulse current info
        ip = 0
        INFO["Pulses"] = {}
        for pulse in self.RAWDataProc.DATADICT["PULSES"]:
            qq = []
            qv = []
            for ipm in range(self.RAWDataProc.DATADICT["nPulseMoments"]):     
                #for istack in self.RAWDataProc.DATADICT["stacks"]:
                #    print ("stack q", self.RAWDataProc.DATADICT[pulse]["Q"][ipm,istack-1])
                qq.append(np.mean(    self.RAWDataProc.DATADICT[pulse]["Q"][ipm,:]) )
                qv.append(np.std(     self.RAWDataProc.DATADICT[pulse]["Q"][ipm,:]/self.RAWDataProc.pulseLength[ip] ))
            qq = np.array(qq)
            qv = np.array(qv)
            iQ = np.argsort(np.array(qq))
            qq = np.array(qq)[iQ]
            qv = np.array(qv)[iQ]
            INFO["Pulses"][pulse] = {}
            INFO["Pulses"][pulse]["units"] = "A"
            INFO["Pulses"][pulse]["current"] = VectorXr(qq/self.RAWDataProc.pulseLength[ip])
            INFO["Pulses"][pulse]["variance"] = VectorXr(qv)
            ip += 1

        # Data
        if self.RAWDataProc.gated == True:
            INFO["Gated"] = {}
            INFO["Gated"]["abscissa units"] = "ms"
            INFO["Gated"]["data units"] = "nT"
            for pulse in self.RAWDataProc.DATADICT["PULSES"]:
                INFO["Gated"][pulse] = {} 
                INFO["Gated"][pulse]["abscissa"] = VectorXr( self.RAWDataProc.GATEDABSCISSA ) 
                INFO["Gated"][pulse]["windows"] = VectorXr( self.RAWDataProc.GATEDWINDOW ) 
                for ichan in self.RAWDataProc.DATADICT[pulse]["chan"]:
                    INFO["Gated"][pulse]["Chan. " + str(ichan)] = {} 
                    #INFO["Gated"][pulse]["Chan. " + str(ichan)]["STD"] =  VectorXr(   np.std(self.RAWDataProc.GATED[ichan]["NR"], axis=0) )
                    INFO["Gated"][pulse]["Chan. " + str(ichan)]["STD"] = VectorXr( np.average(self.RAWDataProc.GATED[ichan]["BN"], axis=0) )
                    for ipm in range(self.RAWDataProc.DATADICT["nPulseMoments"]):     
                        INFO["Gated"][pulse]["Chan. " + str(ichan)]["Q-"+str(ipm) + " CA"] = VectorXr(self.RAWDataProc.GATED[ichan]["CA"][ipm])   
                        INFO["Gated"][pulse]["Chan. " + str(ichan)]["Q-"+str(ipm) + " RE"] = VectorXr(self.RAWDataProc.GATED[ichan]["RE"][ipm])   
                        INFO["Gated"][pulse]["Chan. " + str(ichan)]["Q-"+str(ipm) + " IM"] = VectorXr(self.RAWDataProc.GATED[ichan]["IM"][ipm])   
                        #INFO["Gated"][pulse]["Chan. " + str(ichan)]["Q-"+str(ipm) + " IP"] = VectorXr(self.RAWDataProc.GATED[ichan]["IP"][ipm])   
                        #INFO["Gated"][pulse]["Chan. " + str(ichan)]["Q-"+str(ipm) + " NR"] = VectorXr(self.RAWDataProc.GATED[ichan]["NR"][ipm])   
                        #INFO["Gated"][pulse]["Chan. " + str(ichan)]["Q-"+str(ipm) + " STD" ] = VectorXr(self.RAWDataProc.GATED[ichan]["SIGMA"][ipm])   
            
            # we have gated data 
            # Window edges  
            # Window centres 

        with open(SaveStr, 'w') as outfile:
            #for line in self.logText:
            #    outfile.write(line+"\n")
            yaml.dump(self.YamlNode, outfile)   
            yaml.dump(INFO, outfile) #, default_flow_style=False)   
 
    def SavePreprocess(self):
     
        #if "Saved" not in self.YamlNode.Processing.keys():
        #    self.YamlNode.Processing["Saved"] = []
        #self.YamlNode.Processing["Saved"].append(datetime.datetime.now().isoformat()) 
        #self.Log()
        
        import pickle, os 
        try:
            with open('.akvo.last.path') as f: 
                fpath = f.readline()  
                pass
        except IOError as e:
            fpath = '.'
        
        fdir = os.path.dirname(fpath)
        # Pickle the preprocessed data dictionary 
        SaveStr = QtWidgets.QFileDialog.getSaveFileName(self, "Save as", fdir, r"Pickle (*.dmp)")
 
        spath,filen=os.path.split(str(SaveStr[0]))
        f = open('.akvo.last.path', 'w')
        f.write( str(spath) ) # prompt last file      
        save = open(SaveStr[0], 'wb')

        # Add some extra info 
        INFO = {}
        INFO["pulseType"] = self.RAWDataProc.pulseType
        INFO["prePulseDelay"] = self.RAWDataProc.prePulseDelay
        INFO["interpulseDelay"] = self.RAWDataProc.interpulseDelay
        INFO["transFreq"] = self.RAWDataProc.transFreq
        INFO["pulseLength"] = self.RAWDataProc.pulseLength
        INFO["TuneCapacitance"] = self.RAWDataProc.TuneCapacitance
        INFO["samp"] = self.RAWDataProc.samp
        INFO["nPulseMoments"] = self.RAWDataProc.nPulseMoments
        INFO["deadTime"] = self.RAWDataProc.deadTime
        INFO["transFreq"] = self.RAWDataProc.transFreq
        INFO["headerstr"] = str(self.headerstr)
        INFO["nDAQVersion"] = self.RAWDataProc.nDAQVersion
        INFO["log"] = yaml.dump( self.YamlNode )  

        TXRX = []
        for ir in range(0, self.ui.txRxTable.rowCount() ):
            txrx = []
            for ic in range(0, self.ui.txRxTable.columnCount() ):
                txrx.append( self.ui.txRxTable.item(ir, ic).text() )
            TXRX.append(txrx)
        INFO["TXRX"] = TXRX       

        if "Stacking" in self.YamlNode.Stacking.keys():
            INFO["sigma"] = self.RawDataProc.sigma

        if "Gate integrate" in self.YamlNode.Stacking.keys():
            INFO["GATED"] = self.RAWDataProc.GATED

        print("META SAVE")
        print("INFO log", INFO["log"])

        self.RAWDataProc.DATADICT["INFO"] = INFO 

        pickle.dump(self.RAWDataProc.DATADICT, save)
        save.close()

    # Export XML file suitable for USGS ScienceBase Data Release
    def ExportXML(self):
        """ This is a filler function for use by USGS collaborators 
        """
        return 42

    def OpenPreprocess(self):
        import pickle

        try:
            with open('.akvo.last.path') as f: 
                fpath = f.readline()  
                pass
        except IOError as e:
            fpath = '.'
        
        #filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '.')
        fpath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open preprocessed file', fpath, r"Pickle Files (*.dmp)")[0]
        
        f = open('.akvo.last.path', 'w')
        f.write( str(fpath) ) # prompt last file      

        self.ui.logTextBrowser.clear()
        self.logText = []

        if len(fpath) == 0:
            return 
        
        pfile = open(fpath,'rb')        

        unpickle = pickle.Unpickler(pfile)
        self.connectGMRDataProcessor()
        self.RAWDataProc.DATADICT = unpickle.load()
       
        # This line causes Akvo to crash, if the header file is no longer there. We don't need to load the 
        # file. TODO, need to disable "Load Data" in Load command though, as that is no longer possible.  
        #self.RAWDataProc.readHeaderFile(self.RAWDataProc.DATADICT["INFO"]["headerstr"])
        self.headerstr = self.RAWDataProc.DATADICT["INFO"]["headerstr"]
 
        self.RAWDataProc.pulseType = self.RAWDataProc.DATADICT["INFO"]["pulseType"] 
        self.RAWDataProc.transFreq = self.RAWDataProc.DATADICT["INFO"]["transFreq"] 
        self.RAWDataProc.pulseLength = self.RAWDataProc.DATADICT["INFO"]["pulseLength"] 
        self.RAWDataProc.TuneCapacitance = self.RAWDataProc.DATADICT["INFO"]["TuneCapacitance"] 
        self.RAWDataProc.samp = self.RAWDataProc.DATADICT["INFO"]["samp"] 
        self.RAWDataProc.nPulseMoments = self.RAWDataProc.DATADICT["INFO"]["nPulseMoments"] 
        self.RAWDataProc.deadTime = self.RAWDataProc.DATADICT["INFO"]["deadTime"] 
        self.RAWDataProc.transFreq = self.RAWDataProc.DATADICT["INFO"]["transFreq"]
        self.RAWDataProc.nDAQVersion = self.RAWDataProc.DATADICT["INFO"]["nDAQVersion"]
        #self.RAWDataProc.prePulseDelay = self.RAWDataProc.DATADICT["INFO"]["prePulseDelay"]
        self.RAWDataProc.dt = 1./self.RAWDataProc.samp 

        self.dataChan = self.RAWDataProc.DATADICT[ self.RAWDataProc.DATADICT["PULSES"][0] ]["chan"]

        # Keep backwards compatibility with prior saved pickles???
        #self.ui.logTextBrowser.clear() 
            #self.ui.logTextBrowser.append( yaml.dump(self.YamlNode)) #, default_flow_style=False)  )
            #for a in self.logText:
            #    self.ui.logTextBrowser.append(str(a))
            #self.ui.logTextBrowser
            #self.ui.logTextBrowser.clear()
            #print ( self.RAWDataProc.DATADICT["INFO"]["log"] )

        if "TXRX" in self.RAWDataProc.DATADICT["INFO"].keys():
            TXRX = self.RAWDataProc.DATADICT["INFO"]["TXRX"]
            self.ui.txRxTable.setRowCount( len(TXRX)) 
            for irow, row in enumerate(TXRX):        
                for icol, val in enumerate(row):        
                    pCell = QtWidgets.QTableWidgetItem()
                    pCell.setText( val ) 
                    pCell.setFlags( QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled )
                    self.ui.txRxTable.setItem(irow, icol, pCell)

        
        self.logText = self.RAWDataProc.DATADICT["INFO"]["log"] # YAML 
        parse = yaml.load( self.logText, Loader=yaml.Loader )

        self.YamlNode = AkvoYamlNode( )  #self.logText )
       
        self.YamlNode.Akvo_VERSION = (yaml.load( self.logText, Loader=yaml.Loader )).Akvo_VERSION
        AKVO_VERSION = np.array(self.YamlNode.Akvo_VERSION.split("."), dtype=int)
        if  AKVO_VERSION[0] >= 1 and AKVO_VERSION[1] >= 2 and AKVO_VERSION[2] >= 3:
            self.RAWDataProc.interpulseDelay = self.RAWDataProc.DATADICT["INFO"]["interpulseDelay"]

        self.YamlNode.Import =   OrderedDict(parse.Import)
        self.YamlNode.Processing = list(parse.Processing)
        self.YamlNode.Stacking = OrderedDict(parse.Stacking)
        self.YamlNode.META =     OrderedDict(parse.META)
        
        self.logGUI()
        self.Log()

            #self.ui.logTextBrowser.append( yaml.dump(self.YamlNode)) #, default_flow_style=False)  )
        #except KeyError:
        #    pass
        # Remove "Saved" and "Loaded" from processing flow 
        #if "Loaded" not in self.YamlNode.Processing.keys():
        #    self.YamlNode.Processing["Loaded"] = []
        #self.YamlNode.Processing["Loaded"].append(datetime.datetime.now().isoformat()) 
        #self.Log()

        if "Stacking" in self.YamlNode.Stacking.keys():
            self.RAWDataProc.sigma = self.RAWDataProc.DATADICT["sigma"]

        # check to see if gate integration has been done, and it's ready to send to Merlin
        if "Gate integrate" in self.YamlNode.Stacking.keys():
            self.RAWDataProc.GATED = self.RAWDataProc.DATADICT["GATED"]
            self.ui.actionExport_Preprocessed_Dataset.setEnabled(True)
            #self.ui.plotGI.setEnabled(True)
 
        # If we got this far, enable all the widgets
        self.ui.lcdNumberTauPulse1.setEnabled(True)
        self.ui.lcdNumberNuTx.setEnabled(True)
        self.ui.lcdNumberTuneuF.setEnabled(True)
        self.ui.lcdNumberSampFreq.setEnabled(True)
        self.ui.lcdNumberNQ.setEnabled(True)

        self.ui.headerFileBox.setEnabled(True)
        self.ui.headerFileBox.setChecked( True )
        self.headerBoxShrink() 
        #self.ui.headerBox2.setVisible(True) 
        self.ui.inputRAWParametersBox.setEnabled(False)
        self.ui.loadDataPushButton.setEnabled(True)
        
        # make plots as you import the datasetmost
        self.ui.plotImportCheckBox.setEnabled(True)
        self.ui.plotImportCheckBox.setChecked(True)

        # enable the LCDs
        self.ui.lcdNumberFID1Length.setEnabled(1)
        self.ui.lcdNumberFID2Length.setEnabled(1)
        self.ui.lcdNumberResampFreq.setEnabled(1)
        self.ui.lcdTotalDeadTime.setEnabled(1)

        # enable META tab 
        self.ui.METATab.setEnabled(1)
        self.ui.siteBox.setEnabled(1)
        
        #self.ui.lcdTotalDeadTime.display( 1e3*self.RAWDataProc.DATADICT["INFO"]["deadTime"] )

        self.ui.headerFileTextBrowser.clear( ) 
        self.ui.headerFileTextBrowser.append( self.RAWDataProc.DATADICT["INFO"]["headerstr"] )
        
        if u"Pulse 1" in self.RAWDataProc.DATADICT.keys():
            self.ui.lcdNumberFID1Length.display(self.RAWDataProc.DATADICT["Pulse 1"]["TIMES"][-1]- self.RAWDataProc.DATADICT["Pulse 1"]["TIMES"][0])
            self.ui.lcdTotalDeadTime.display( round(1e3*(self.RAWDataProc.DATADICT["Pulse 1"]["TIMES"][0]-self.RAWDataProc.DATADICT["Pulse 1"]["PULSE_TIMES"][-1]), 3) )
            #print("CALC DEAD",    (1e3*(self.RAWDataProc.prePulseDelay))) # -  (self.RAWDataProc.DATADICT["Pulse 1"]["TIMES"][0]-self.RAWDataProc.DATADICT["Pulse 1"]["PULSE_TIMES"][-1])), 3) )
        if u"Pulse 2" in self.RAWDataProc.DATADICT.keys():
            self.ui.lcdNumberFID1Length.display(self.RAWDataProc.DATADICT["Pulse 2"]["TIMES"][-1]- self.RAWDataProc.DATADICT["Pulse 2"]["TIMES"][0])
            self.ui.lcdTotalDeadTime.display( 1e3 * (self.RAWDataProc.DATADICT["Pulse 2"]["TIMES"][0]-self.RAWDataProc.DATADICT["Pulse 2"]["PULSE_TIMES"][-1]) )
 
        # Update info from the header into the GUI
        self.ui.pulseTypeTextBrowser.clear()
        self.ui.pulseTypeTextBrowser.append(self.RAWDataProc.pulseType)
        self.ui.lcdNumberNuTx.display(self.RAWDataProc.transFreq)
        self.ui.lcdNumberTauPulse1.display(1e3*self.RAWDataProc.pulseLength[0])
        self.ui.lcdNumberTuneuF.display(self.RAWDataProc.TuneCapacitance)
        self.ui.lcdNumberResampFreq.display(self.RAWDataProc.samp)
        self.ui.lcdNumberSampFreq.display(50000) # TODO, if non GMR is supported, query
        self.ui.lcdNumberNQ.display(self.RAWDataProc.nPulseMoments)
        self.ui.DeadTimeSpinBox.setValue(1e3*self.RAWDataProc.deadTime)
        self.ui.CentralVSpinBox.setValue( self.RAWDataProc.transFreq )
       
        if self.RAWDataProc.pulseType != "FID":
            self.ui.lcdNumberTauPulse2.setEnabled(1)
            self.ui.lcdNumberTauPulse2.display(1e3*self.RAWDataProc.pulseLength[1])
            self.ui.lcdNumberTauDelay.setEnabled(1)
            self.ui.lcdNumberTauDelay.display(1e3*self.RAWDataProc.interpulseDelay)

        self.ui.FIDProcComboBox.clear() 
        if self.RAWDataProc.pulseType == "4PhaseT1" or self.RAWDataProc.pulseType == "T1":
            self.ui.FIDProcComboBox.insertItem(0, "Pulse 1") #, const QVariant & userData = QVariant() )
            self.ui.FIDProcComboBox.insertItem(1, "Pulse 2") #, const QVariant & userData = QVariant() )
            self.ui.FIDProcComboBox.insertItem(2, "Both")    #, const QVariant & userData = QVariant() )
            if len( self.RAWDataProc.DATADICT["PULSES"]) == 2:
                self.ui.FIDProcComboBox.setCurrentIndex (2)
            elif  self.RAWDataProc.DATADICT["PULSES"][0] == "Pulse 1":
                self.ui.FIDProcComboBox.setCurrentIndex (0)
            else:
                self.ui.FIDProcComboBox.setCurrentIndex (1)

        elif self.RAWDataProc.pulseType == "FID":
            self.ui.FIDProcComboBox.insertItem(0, "Pulse 1") #, const QVariant & userData = QVariant() )
            self.ui.FIDProcComboBox.setCurrentIndex (0)

#         QtCore.QObject.connect(self.RAWDataProc, QtCore.SIGNAL("updateProgress(int)"), self.updateProgressBar)
#         QtCore.QObject.connect(self.RAWDataProc, QtCore.SIGNAL("enableDSP()"), self.enableDSP)
#         QtCore.QObject.connect(self.RAWDataProc, QtCore.SIGNAL("doneStatus()"), self.doneStatus)
        self.RAWDataProc.progressTrigger.connect(self.updateProgressBar)
        self.RAWDataProc.enableDSPTrigger.connect(self.enableDSP)
        self.RAWDataProc.doneTrigger.connect(self.doneStatus)
        
        self.enableAll()
 
    def loadRAW(self):

        #################################################
        # Check to make sure we are ready to process

        # Header
        if self.RAWDataProc == None:
            err_msg = "You need to load a header first."
            reply = QtGui.QMessageBox.critical(self, 'Error', 
                err_msg) #, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
            return
        
        # Stacks 
        try:
            self.procStacks = np.array(eval(str("np.r_["+self.ui.stacksLineEdit.text())+"]"))
        except:
            err_msg = "You need to set your stacks correctly.\n" + \
                      "This should be a Python Numpy interpretable list\n" + \
                      "of stack indices. For example 1:24 or 1:4,8:24"
            QtGui.QMessageBox.critical(self, 'Error', err_msg) 
            return

        # Data Channels
        #Chan = np.arange(0,9,1)
        try:
            self.dataChan = np.array(eval(str("np.r_["+self.ui.dataChanLineEdit.text())+"]"))
        except:
            #QMessageBox messageBox;
            #messageBox.critical(0,"Error","An error has occured !");
            #messageBox.setFixedSize(500,200);
            #quit_msg = "Are you sure you want to exit the program?"
            #reply = QtGui.QMessageBox.question(self, 'Message', 
            #    quit_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
            err_msg = "You need to set your data channels correctly.\n" + \
                      "This should be a Python Numpy interpretable list\n" + \
                      "of indices. For example 1 or 1:3 or 1:3 5\n\n" + \
                      "valid GMR data channels fall between 1 and 8. Note that\n" +\
                      "1:3 is not inclusive of 3 and is the same as 1,2 "
            reply = QtGui.QMessageBox.critical(self, 'Error', 
                err_msg) #, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
            return
        #############################
        # Reference Channels
        # TODO make sure no overlap between data and ref channels
        self.refChan = np.array( () )
        if str(self.ui.refChanLineEdit.text()): # != "none":
            try:
                self.refChan = np.array(eval(str("np.r_["+self.ui.refChanLineEdit.text())+"]"))
            except:
                err_msg = "You need to set your reference channels correctly.\n" + \
                      "This should be a Python Numpy interpretable list\n" + \
                      "of indices. For example 1 or 1:3 or 1:3 5\n\n" + \
                      "valid GMR data channels fall between 1 and 8. Note that\n" +\
                      "1:3 is not inclusive of 3 and is the same as 1,2 "
                QtGui.QMessageBox.critical(self, 'Error', err_msg) 
                return

        #####################################################
        # Load data

        self.lock("loading RAW GMR dataset")

        if self.RAWDataProc.pulseType == "FID":
            self.procThread = thread.start_new_thread(self.RAWDataProc.loadFIDData, \
                (str(self.headerstr), self.procStacks, self.dataChan, self.refChan, \
                 str(self.ui.FIDProcComboBox.currentText()), self.ui.mplwidget, \
                1e-3 * self.ui.DeadTimeSpinBox.value( ), self.ui.plotImportCheckBox.isChecked() )) #, self)) 
        elif self.RAWDataProc.pulseType == "4PhaseT1":
            self.procThread = thread.start_new_thread(self.RAWDataProc.load4PhaseT1Data, \
                (str(self.headerstr), self.procStacks, self.dataChan, self.refChan, \
                 str(self.ui.FIDProcComboBox.currentText()), self.ui.mplwidget, \
                1e-3 * self.ui.DeadTimeSpinBox.value( ), self.ui.plotImportCheckBox.isChecked() )) #, self)) 
        elif self.RAWDataProc.pulseType == "T1":
            self.procThread = thread.start_new_thread(self.RAWDataProc.loadT1Data, \
                (str(self.headerstr), self.procStacks, self.dataChan, self.refChan, \
                 str(self.ui.FIDProcComboBox.currentText()), self.ui.mplwidget, \
                1e-3 * self.ui.DeadTimeSpinBox.value( ), self.ui.plotImportCheckBox.isChecked() )) #, self)) 
            #self.procThread = thread.start_new_thread(self.RAWDataProc.load4PhaseT1Data, \
            #    (str(self.headerstr), self.procStacks, self.dataChan, self.refChan, \
            #     str(self.ui.FIDProcComboBox.currentText()), self.ui.mplwidget, \
            #    1e-3 * self.ui.DeadTimeSpinBox.value( ), self.ui.plotImportCheckBox.isChecked() )) #, self)) 

        self.YamlNode.Import["GMR Header"] = self.headerstr
        self.YamlNode.Import["opened"] = datetime.datetime.now().isoformat() 
        self.YamlNode.Import["pulse Type"] = str(self.RAWDataProc.pulseType) 
        self.YamlNode.Import["stacks"] = self.procStacks.tolist() 
        self.YamlNode.Import["data channels"] = self.dataChan.tolist()  
        self.YamlNode.Import["reference channels"] = self.refChan.tolist() 
        self.YamlNode.Import["pulse records"] = str(self.ui.FIDProcComboBox.currentText())  
        self.YamlNode.Import["instrument dead time"] = (1e-3 * self.ui.DeadTimeSpinBox.value( ))    

        self.Log (  )     
        
        # enable META tab 
        self.ui.METATab.setEnabled(1)
        self.ui.siteBox.setEnabled(1)

        # should be already done
#        QtCore.QObject.connect(self.RAWDataProc, QtCore.SIGNAL("updateProgress(int)"), self.updateProgressBar)
#        QtCore.QObject.connect(self.RAWDataProc, QtCore.SIGNAL("enableDSP()"), self.enableDSP)
#        QtCore.QObject.connect(self.RAWDataProc, QtCore.SIGNAL("doneStatus()"), self.doneStatus)

        #self.ui.ProcessedBox.setEnabled(True)
        self.ui.lcdNumberFID1Length.setEnabled(1)
        self.ui.lcdNumberFID2Length.setEnabled(1)
        self.ui.lcdNumberResampFreq.setEnabled(1)
        self.ui.lcdTotalDeadTime.setEnabled(1)

        self.ui.lcdTotalDeadTime.display( self.ui.DeadTimeSpinBox.value( ) )
        #self.ui.lcdTotalDeadTime.display( round(1e3*(self.RAWDataProc.DATADICT["Pulse 1"]["TIMES"][0]-self.RAWDataProc.DATADICT["Pulse 1"]["PULSE_TIMES"][-1]), 3) )
        
        #self.ui.lcdNumberFID1Length.display(0)
        #self.ui.lcdNumberFID2Length.display(0)
        #self.ui.lcdNumberResampFreq.display( self.RAWDataProc.samp )
 
        self.mpl_toolbar = NavigationToolbar2QT(self.ui.mplwidget, self.ui.mplwidget)
        self.ui.mplwidget.draw()

    def Log(self):
        #for line in yaml.dump(self.YamlNode, default_flow_style=False):
        #for line in nlogText: 
        #    self.ui.logTextBrowser.append( line )
        #    self.logText.append( line )
        self.ui.logTextBrowser.clear()
        self.ui.logTextBrowser.append( yaml.dump(self.YamlNode )) 

    def disable(self):
        self.ui.inputRAWParametersBox.setEnabled(False)
        self.ui.BandPassBox.setEnabled(False)
        self.ui.downSampleGroupBox.setEnabled(False)
        self.ui.windowFilterGroupBox.setEnabled(False)
        self.ui.harmonicBox.setEnabled(False)
#        self.ui.despikeGroupBox.setEnabled(False)
        self.ui.adaptBox.setEnabled(False)
        self.ui.adaptFDBox.setEnabled(False)
        self.ui.qCalcGroupBox.setEnabled(False)    
        self.ui.FDSmartStackGroupBox.setEnabled(False)    
        self.ui.sumDataBox.setEnabled(False)    
        self.ui.qdGroupBox.setEnabled(False)
        self.ui.gateBox.setEnabled(False)

    def enableAll(self):
        self.enableDSP()
        self.enableQC()

    def enableDSP(self):

        # Bandpass filter
        self.ui.BandPassBox.setEnabled(True)
        self.ui.BandPassBox.setChecked(True)
        self.ui.bandPassGO.setEnabled(False) # need to design first
        self.ui.plotBP.setEnabled(True)
        self.ui.plotBP.setChecked(True)

        # downsample 
        self.ui.downSampleGroupBox.setEnabled(True)
        self.ui.downSampleGroupBox.setChecked(True)
       
        # window
        self.ui.windowFilterGroupBox.setEnabled(True)
        self.ui.windowFilterGroupBox.setChecked(True)
 
        # Despike
#        self.ui.despikeGroupBox.setEnabled(True)
#        self.ui.despikeGroupBox.setChecked(False)

        # Adaptive filtering 
        self.ui.adaptBox.setEnabled(True)
        self.ui.adaptBox.setChecked(True)
        
        # FD Adaptive filtering 
        self.ui.adaptFDBox.setEnabled(True)
        self.ui.adaptFDBox.setChecked(False)
        
        # Harmonic
        self.ui.harmonicBox.setEnabled(True)
        self.ui.harmonicBox.setChecked(True)
        self.LCDHarmonics()
        self.LCDHarmonics2()

        # sum group box
        try:
            if len(self.dataChan) > 1:
                self.ui.sumDataBox.setEnabled(True)
                self.ui.sumDataBox.setChecked(False)
        except:
            pass

        # Quadrature Detect
        self.ui.qdGroupBox.setEnabled(True)
        self.ui.qdGroupBox.setChecked(True)

        self.enableQC() 

    def enableQC(self):
        
        # Q calc
        self.ui.qCalcGroupBox.setEnabled(True)    
        self.ui.qCalcGroupBox.setChecked(True)    

        # FD SmartStack
        self.ui.FDSmartStackGroupBox.setEnabled(True)    
        self.ui.FDSmartStackGroupBox.setChecked(True)    
        
        # Quadrature detect
        try:
            for pulse in self.RAWDataProc.DATADICT["PULSES"]: 
                np.shape(self.RAWDataProc.DATADICT[pulse]["Q"])
            self.RAWDataProc.DATADICT["stack"] 
            self.ui.qdGroupBox.setEnabled(True)   
            self.ui.qdGroupBox.setChecked(True)    
        except:
            self.ui.qdGroupBox.setEnabled(False)    
            self.ui.qdGroupBox.setChecked(False)    
        
        # Gating
        try:
            self.RAWDataProc.DATADICT["CA"] 
            self.ui.gateBox.setEnabled(True)
            self.ui.gateBox.setChecked(True)
        except:
            self.ui.gateBox.setEnabled(False)
            self.ui.gateBox.setChecked(False)

    def despikeFilter(self):
        self.lock("despike filter")
        thread.start_new_thread(self.RAWDataProc.despike, \
                (self.ui.windowSpinBox.value(), \
                self.ui.thresholdSpinBox.value(), \
                str(self.ui.replComboBox.currentText()), \
                self.ui.rollOnSpinBox.value(), \
                self.ui.despikeInterpWinSpinBox.value(),
                self.ui.mplwidget))

    def calcQ(self):
        if "Calc Q" not in self.YamlNode.Stacking.keys():
            #print("In CalcQ", yaml.dump(self.YamlNode.Processing)  )
            self.YamlNode.Stacking["Calc Q"] = True
            #print( yaml.dump(self.YamlNode.Processing)  )
            self.Log()
        else:
            err_msg = "Q values have already been calculated"
            reply =QtWidgets.QMessageBox.critical(self, 'Error', 
                err_msg) 
            return 

        self.lock("pulse moment calculation")
        thread.start_new_thread(self.RAWDataProc.effectivePulseMoment, \
                (self.ui.CentralVSpinBox.value(), \
                self.ui.mplwidget))

    def harmonicModel(self):
        
        self.lock("harmonic noise modelling")
        
        Harm = OrderedDict()
        Harm["STEP " + str(len(self.YamlNode.Processing))] = "Harmonic modelling"
        Harm["NF"] = str( self.ui.NHarmonicsFreqsSpin.value() ) 
        Harm["Segments"] = str( self.ui.NSegments.value() ) 
        Harm["Proc. ref."] = self.ui.harmRef.isChecked() 
        
        
        if self.ui.searchAll.currentText() == "All":
            Harm["search"] = self.ui.searchAll.currentText()
            Search = False  
        else:
            #Harm["search"] = self.ui.searchAll.currentText() 
            Harm["search"] = str(self.ui.Nsearch.value()) 
            Search = self.ui.Nsearch.value() 
        if self.ui.boundsCheck.isChecked():
            Harm["Bounds"] = str(self.ui.bounds.value()) 
            Bounds = self.ui.bounds.value() 
        else:
            Harm["Bounds"] = self.ui.boundsCheck.isChecked() 
            Bounds = 0

        Harm["f0K1"] = str( self.ui.f0K1Spin.value() )
        Harm["f0KN"] = str( self.ui.f0KNSpin.value() )
        Harm["f0Ks"] = str( self.ui.f0KsSpin.value() )
        Harm["f0"] = str( self.ui.f0Spin.value() )
        if self.ui.NHarmonicsFreqsSpin.value() > 1:
            Harm["f1K1"] = str( self.ui.f1K1Spin.value() )
            Harm["f1KN"] = str( self.ui.f1KNSpin.value() )
            Harm["f1Ks"] = str( self.ui.f1KsSpin.value() )
            Harm["f1"] = str( self.ui.f1Spin.value() )
        self.YamlNode.Processing.append(Harm)
        self.Log()

        thread.start_new_thread(self.RAWDataProc.harmonicModel, \
                ( \
                 self.ui.NHarmonicsFreqsSpin.value(), \
                 self.ui.f0Spin.value(), \
                 self.ui.f0K1Spin.value(), \
                 self.ui.f0KNSpin.value(), \
                 self.ui.f0KsSpin.value(), \
                 self.ui.NSegments.value(), \
                 self.ui.f1Spin.value(), \
                 self.ui.f1K1Spin.value(), \
                 self.ui.f1KNSpin.value(), \
                 self.ui.f1KsSpin.value(), \
                 Search, \
                 Bounds, \
                 self.ui.harmRef.isChecked(), \
                 self.ui.plotHarmonic.isChecked(), \
                 self.ui.mplwidget \
                ) \
        )

    def FDSmartStack(self):

        if "TD stack" not in self.YamlNode.Stacking.keys():
            self.YamlNode.Stacking["TD stack"] = {}
            self.YamlNode.Stacking["TD stack"]["outlier"] = str( self.ui.outlierTestCB.currentText() ) 
            self.YamlNode.Stacking["TD stack"]["cutoff"] = str( self.ui.MADCutoff.value() )
            self.Log()
        else:
            err_msg = "TD noise cancellation has already been applied!"
            reply =QtWidgets.QMessageBox.critical(self, 'Error', 
                err_msg) 
            return 
        
        self.lock("time-domain smart stack")
        thread.start_new_thread(self.RAWDataProc.TDSmartStack, \
                (str(self.ui.outlierTestCB.currentText()), \
                self.ui.MADCutoff.value(),
                self.ui.mplwidget))

    def adaptFilter(self):
        
        self.lock("TD noise cancellation filter")
        
        # Log processing 
        Adapt = OrderedDict()
        Adapt["STEP " + str(len(self.YamlNode.Processing)) ] = "TD noise cancellation"
        #print(Adapt) # this locks STEP in as first...
        Adapt["n_Taps"] = self.ui.MTapsSpinBox.value()
        Adapt["lambda"] = self.ui.adaptLambdaSpinBox.value()
        Adapt["truncate"] = self.ui.adaptTruncateSpinBox.value()
        Adapt["mu"] = self.ui.adaptMuSpinBox.value()
        Adapt["PCA"] = self.ui.PCAComboBox.currentText()
        #Adapt # this locsk in the dict ordering...
        #print(Adapt) # this locks in the dict...
        self.YamlNode.Processing.append(Adapt)
        self.Log( )

        thread.start_new_thread(self.RAWDataProc.adaptiveFilter, \
                (self.ui.MTapsSpinBox.value(), \
                self.ui.adaptLambdaSpinBox.value(), \
                self.ui.adaptTruncateSpinBox.value(), \
                self.ui.adaptMuSpinBox.value(), \
                str(self.ui.PCAComboBox.currentText()), \
                self.ui.mplwidget))

    def sumDataChans(self): 

        self.lock("Summing data channels")
        
        Sum = OrderedDict()
        Sum["STEP " + str(len(self.YamlNode.Processing))] = "Channel sum"
        self.YamlNode.Processing.append(Sum)
        self.Log( )

        self.dataChan = [self.dataChan[0]]
        self.ui.sumDataBox.setEnabled(False)    
        thread.start_new_thread( self.RAWDataProc.sumData, ( self.ui.mplwidget, 7 ) )
 
    def adaptFilterFD(self):
        self.lock("FD noise cancellation filter")
        thread.start_new_thread(self.RAWDataProc.adaptiveFilterFD, \
                (str(self.ui.windowTypeComboBox.currentText()), \
                self.ui.windowBandwidthSpinBox.value(), \
                self.ui.CentralVSpinBox.value(), \
                self.ui.mplwidget))

    def logGUI(self):
        # You have a race condiditon where the GUI is modifying the Yaml node while you are updating it
        # hence, we need to cache these. More elegant solutions exist       
        if "B_0" in self.YamlNode.META:
            B0 = self.YamlNode.META["B_0"]["intensity"]
            Bdec = self.YamlNode.META["B_0"]["dec"]
            Binc = self.YamlNode.META["B_0"]["inc"]
        if "DateTime" in self.YamlNode.META:
            [Date,Time] = self.YamlNode.META["DateTime"].split("T")
            year,month,day = Date.split("-")
        if "Temp" in self.YamlNode.META:
            temp = float(self.YamlNode.META["Temp"])
        if "Coordinates" in self.YamlNode.META:
            UTM = self.YamlNode.META["Coordinates"]["UTM"] 
            LatBand = self.YamlNode.META["Coordinates"]["LatBand"] 
            Ellipsoid = self.YamlNode.META["Coordinates"]["ellipsoid"] 

        # and set
        if "Location" in self.YamlNode.META:
            self.ui.locEdit.setText( self.YamlNode.META["Location"] )
        if "Field Notes" in self.YamlNode.META:
            self.ui.txtComments.setText( self.YamlNode.META["Field Notes"] )
        if "B_0" in self.YamlNode.META:
            self.ui.incSpinBox.setValue( Binc )  
            self.ui.decSpinBox.setValue( Bdec )  
            self.ui.intensitySpinBox.setValue( B0 ) 
        if "DateTime" in self.YamlNode.META:
            self.ui.dateEdit.setDate( datetime.date(int(year), int(month), int(day)) )            
            self.ui.timeEdit.setTime( datetime.time.fromisoformat( Time ) )            
        if "Temp" in self.YamlNode.META:
            self.ui.tempSpinBox.setValue(temp)
        if "Coordinates" in self.YamlNode.META:
            self.ui.UTMzone.setCurrentText(UTM)
            self.ui.latBand.setCurrentText(LatBand) 
            self.ui.ellipsoid.setCurrentText(Ellipsoid)


    def logSite(self):
        self.YamlNode.META["Location"] = self.ui.locEdit.text()
        self.YamlNode.META["Coordinates"] = OrderedDict() 
        self.YamlNode.META["Coordinates"]["UTM"] = self.ui.UTMzone.currentText()
        self.YamlNode.META["Coordinates"]["LatBand"] = self.ui.latBand.currentText() 
        self.YamlNode.META["Coordinates"]["ellipsoid"] =  self.ui.ellipsoid.currentText()
        self.YamlNode.META["DateTime"] = self.ui.dateEdit.date().toString("yyyy-MM-dd") + "T" + str( self.ui.timeEdit.time().toString("hh:mm") )
        self.YamlNode.META["Temp"] = self.ui.tempSpinBox.value()
        self.YamlNode.META["B_0"] = OrderedDict()
        self.YamlNode.META["B_0"]["inc"] = self.ui.incSpinBox.value()
        self.YamlNode.META["B_0"]["dec"] = self.ui.decSpinBox.value()
        self.YamlNode.META["B_0"]["intensity"] = self.ui.intensitySpinBox.value()
        self.YamlNode.META["Field Notes"] = self.ui.txtComments.toPlainText()

        self.YamlNode.META["Loops"] = OrderedDict()
        for loop in self.loops:
            print(self.loops[loop])
            self.YamlNode.META["Loops"][loop] = loop + ".pwa" 

        self.Log()

    def bandPassFilter(self):
        
        self.lock("bandpass filter")        
        
        # Log processing 
        Band = OrderedDict()
        Band["STEP " + str(len(self.YamlNode.Processing))] = "Bandpass filter"
        Band["central_nu"] = str(self.ui.CentralVSpinBox.value())
        Band["passband"] = str(self.ui.passBandSpinBox.value())
        Band["stopband"] = str(self.ui.stopBandSpinBox.value()) 
        Band["gpass"] = str(self.ui.gpassSpinBox.value())
        Band["gstop"] = str(self.ui.gstopSpinBox.value())
        Band["type"] = str(self.ui.fTypeComboBox.currentText())
        self.YamlNode.Processing.append(Band)
        self.Log( )

        nv = self.ui.lcdTotalDeadTime.value( ) + self.ui.lcdNumberFTauDead.value()
        self.ui.lcdTotalDeadTime.display( nv )
        thread.start_new_thread(self.RAWDataProc.bandpassFilter, \
                (self.ui.mplwidget, 0, self.ui.plotBP.isChecked() ))

    def downsample(self):

        self.lock("resampling")

        # Log processing 
        Resample = OrderedDict() 
        Resample["STEP "+ str(len(self.YamlNode.Processing))] = "Resample"
        Resample["downsample factor"] = str(self.ui.downSampleSpinBox.value())
        Resample["truncate length"] = str(self.ui.truncateSpinBox.value()) 
        self.YamlNode.Processing.append(Resample)
        self.Log( )
        
        thread.start_new_thread(self.RAWDataProc.downsample, \
                (self.ui.truncateSpinBox.value(), \
                self.ui.downSampleSpinBox.value(), \
                self.ui.dsPlot.isChecked(), \
                self.ui.mplwidget))

    def quadDet(self):

        method = ['trf','dogbox','lm'][int(self.ui.QDMethod.currentIndex())]
        loss = ['linear','soft_l1','cauchy','huber'][int(self.ui.QDLoss.currentIndex())]         

        # allow overwrite of Quad Det.    
        self.YamlNode.Stacking["Quadrature detection"] = {}
        self.YamlNode.Stacking["Quadrature detection"]["trim"] = str( self.ui.trimSpin.value() )
        self.YamlNode.Stacking["Quadrature detection"]["method"] = method 
        self.YamlNode.Stacking["Quadrature detection"]["loss"] = loss
        self.Log()

        #if "Quadrature detection" not in self.YamlNode.Processing.keys():
        #    self.YamlNode.Processing["Quadrature detection"] = {}
        #    self.YamlNode.Processing["Quadrature detection"]["trim"] = str( self.ui.trimSpin.value() )
        #    self.Log()
        #else:
        #    self.YamlNode.Processing["Quadrature detection"] = {}
        #    self.YamlNode.Processing["Quadrature detection"]["trim"] = str( self.ui.trimSpin.value() )
        #    self.Log()
            #err_msg = "Quadrature detection has already been done!"
            #reply =QtWidgets.QMessageBox.critical(self, 'Error', 
            #    err_msg) 
            #return 
        self.lock("quadrature detection")
        thread.start_new_thread(self.RAWDataProc.quadDet, \
                (self.ui.trimSpin.value(), method, loss, self.ui.mplwidget))

        self.ui.plotQD.setEnabled(True)
    
    def plotQD(self):
        self.lock("plot QD")

        thread.start_new_thread(self.RAWDataProc.plotQuadDet, \
                (self.ui.trimSpin.value(), int(self.ui.QDType.currentIndex()), self.ui.mplwidget))


    def gateIntegrate(self):
        
        if "Gate integrate" not in self.YamlNode.Stacking.keys():
            self.YamlNode.Stacking["Gate integrate"] = {}
        self.YamlNode.Stacking["Gate integrate"]["gpd"] = str(self.ui.GPDspinBox.value( ) )
        self.Log()
 
        self.lock("gate integration")
        thread.start_new_thread(self.RAWDataProc.gateIntegrate, \
                (self.ui.GPDspinBox.value(), self.ui.trimSpin.value(), self.ui.mplwidget))
        
        self.ui.actionExport_Preprocessed_Dataset.setEnabled(True)
        self.ui.plotGI.setEnabled(True)

    def plotGI(self):
        self.lock("plot gate integrate")
        thread.start_new_thread(self.RAWDataProc.plotGateIntegrate, \
                (self.ui.GPDspinBox.value(), self.ui.trimSpin.value(), \
                self.ui.QDType_2.currentIndex(),  self.ui.mplwidget))
 
    def designFilter(self):
        [bord, fe] = self.RAWDataProc.designFilter( \
                self.ui.CentralVSpinBox.value(), \
                self.ui.passBandSpinBox.value(), \
                self.ui.stopBandSpinBox.value(), \
                self.ui.gpassSpinBox.value(), \
                self.ui.gstopSpinBox.value(), \
                str(self.ui.fTypeComboBox.currentText()),
                self.ui.mplwidget
        )
        self.ui.lcdNumberFilterOrder.display(bord)
        self.ui.lcdNumberFTauDead.display(1e3*fe)
        self.ui.bandPassGO.setEnabled(1)

        ################################################################
        # Hack for MacOS to force refresh of group box and plot
        # this has an undesirable effect that it causes the groupbox to 'jump' up
        # TODO come up with a better solution  
        self.ui.mplwidget.hide()
        self.ui.mplwidget.show()
        self.ui.BandPassBox.hide()
        self.ui.BandPassBox.show()
 
    def windowFilter(self):
        
        self.lock("window filter")
        
        # Log processing 
        Window = OrderedDict()
        Window["STEP " + str(len(self.YamlNode.Processing))] = "Window filter"
        Window["type"] = str(self.ui.windowTypeComboBox.currentText()) 
        Window["width"] = str(self.ui.windowBandwidthSpinBox.value()) 
        Window["centre"] = str(self.ui.CentralVSpinBox.value() )
        Window["trim"] = str(self.ui.windowTrim.isChecked())
        self.YamlNode.Processing.append(Window)
        self.Log( )
        
        if self.ui.windowTrim.isChecked():
            nv = self.ui.lcdTotalDeadTime.value( ) + self.ui.lcdWinDead.value()
            self.ui.lcdTotalDeadTime.display( nv )
        
        thread.start_new_thread(self.RAWDataProc.windowFilter, \
                (str(self.ui.windowTypeComboBox.currentText()), \
                self.ui.windowBandwidthSpinBox.value(), \
                self.ui.CentralVSpinBox.value(), \
                self.ui.windowTrim.isChecked(), \
                self.ui.mplwidget))

    def designFDFilter(self):

        mPulse = "None"
        if u"Pulse 1" in self.RAWDataProc.DATADICT.keys():
            mPulse = u"Pulse 1"
        elif u"Pulse 2" in self.RAWDataProc.DATADICT.keys():
            mPulse = u"Pulse 2"
        a,b,c,d,dead,ndead = self.RAWDataProc.computeWindow( \
                mPulse,
                self.ui.windowBandwidthSpinBox.value(), \
                self.ui.CentralVSpinBox.value(), \
                str(self.ui.windowTypeComboBox.currentText()), \
                self.ui.mplwidget )

        self.ui.lcdWinDead.display(dead)

        ################################################################
        # Hack for MacOS to force refresh of group box and plot
        # this has an undesirable effect that it causes the groupbox to 'jump' up
        # TODO come up with a better solution  
        self.ui.mplwidget.hide()
        self.ui.mplwidget.show()
        self.ui.windowFilterGroupBox.hide()
        self.ui.windowFilterGroupBox.show()

    def updateProgressBar(self, percent):
        self.ui.barProgress.setValue(percent)

    def updateProc(self):
        if   str(self.ui.FIDProcComboBox.currentText()) == "Pulse 1":
            self.ui.lcdNumberFID1Length.display(self.RAWDataProc.DATADICT["Pulse 1"]["TIMES"][-1]- self.RAWDataProc.DATADICT["Pulse 1"]["TIMES"][0])
        elif str(self.ui.FIDProcComboBox.currentText()) == "Pulse 2":
            self.ui.lcdNumberFID2Length.display(self.RAWDataProc.DATADICT["Pulse 2"]["TIMES"][-1]- self.RAWDataProc.DATADICT["Pulse 2"]["TIMES"][0])
        else:
            self.ui.lcdNumberFID1Length.display(self.RAWDataProc.DATADICT["Pulse 1"]["TIMES"][-1]- self.RAWDataProc.DATADICT["Pulse 1"]["TIMES"][0])
            self.ui.lcdNumberFID2Length.display(self.RAWDataProc.DATADICT["Pulse 2"]["TIMES"][-1]- self.RAWDataProc.DATADICT["Pulse 2"]["TIMES"][0])
        self.ui.lcdNumberResampFreq.display( self.RAWDataProc.samp )
    
    def doneStatus(self): # unlocks GUI
        self.ui.statusbar.clearMessage ( )
        self.ui.barProgress.hide()
        self.updateProc()       
        self.enableAll()

    def lock(self, string):
        self.ui.statusbar.showMessage ( string )
        self.ui.barProgress.show()
        self.ui.barProgress.setValue(0)
        self.disable()

    def unlock(self):
        self.ui.statusbar.clearMessage ( )
        self.ui.barProgress.hide()
        self.enableAll()

    def done(self):
        self.ui.statusbar.showMessage ( "" )


################################################################
################################################################

# Boiler plate main function
import pkg_resources
from pkg_resources import resource_string
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from akvo.gui.logo import plotLogo

def main():
    
    # splash screen logo 
    logo = pkg_resources.resource_filename(__name__, 'akvo.png')
    logo2 = pkg_resources.resource_filename(__name__, 'akvo2.png')
    qApp = QtWidgets.QApplication(sys.argv)

    ssplash = True
    if ssplash:
        pixmap = QtGui.QPixmap(logo)
        splash = QtWidgets.QSplashScreen(pixmap, QtCore.Qt.WindowStaysOnTopHint)
        splash.show()
    
    aw = ApplicationWindow()
    #img=mpimg.imread(logo)
    for ax in [ aw.ui.mplwidget ]: 
        ax.fig.clear()
        subplot = ax.fig.add_subplot(211)
        # old logo plot
        ax.fig.patch.set_facecolor( None )
        ax.fig.patch.set_alpha( .0 )
        #subplot.imshow(img)
        #ax.fig.patch.set_visible(False)
        subplot.axis('off')
        
        plotLogo(subplot) 
        subplot.xaxis.set_major_locator(plt.NullLocator()) 
        subplot.yaxis.set_major_locator(plt.NullLocator()) 
        
        subplot2 = ax.fig.add_subplot(212)
        subplot2.text(0.5, 1.,'surface NMR workbench',
            horizontalalignment='center',
            verticalalignment='center',
            size=22,
            transform = subplot2.transAxes)
        subplot2.xaxis.set_major_locator(plt.NullLocator()) 
        subplot2.yaxis.set_major_locator(plt.NullLocator()) 
        subplot2.axis('off')
        
        ax.draw()

    if ssplash:
        splash.showMessage("Loading modules")
        splash.finish(aw)
        #time.sleep(1) 

    aw.setWindowTitle("Akvo v"+str(VERSION)) 
    aw.show()
    qApp.setWindowIcon(QtGui.QIcon(logo2))
    sys.exit(qApp.exec_())

if __name__ == "__main__":
    main()
