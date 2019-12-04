from PyQt5 import uic
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QTextEdit, QApplication, QDialog

def p(x):
    print (x)

class callScript(QDialog):

    #def __init__(self):
    #    super().__init__()
    
    def setupCB(self, akvoData, kernelParams, SaveStr):

        #QtGui.QWidget.__init__(self)
        #uic.loadUi('redirect.ui', self)

        #print ('Connecting process')
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdoutReady)
        self.process.readyReadStandardError.connect(self.stderrReady)
        self.process.started.connect(lambda: p('Started!'))
        self.process.finished.connect(lambda: p('Finished!'))

        #print ('Starting process')
        #self.process.start('python', ['calcAkvoKernel.py', akvoData, TxCoil, SaveStr])
        self.process.start('akvoK0', [ akvoData, kernelParams, SaveStr])

    def setupQTInv(self, params):

        #QtGui.QWidget.__init__(self)
        #uic.loadUi('redirect.ui', self)

        #print ('Connecting process')
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdoutReady)
        self.process.readyReadStandardError.connect(self.stderrReady)
        self.process.started.connect(lambda: p('Started!'))
        self.process.finished.connect(lambda: p('Finished!'))

        #print ('Starting process')
        #self.process.start('python', ['calcAkvoKernel.py', akvoData, TxCoil, SaveStr])
        self.process.start('akvoQT', [params])

    def append(self, text):
        cursor = self.ui.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.ui.textEdit.ensureCursorVisible()
        #MyTextEdit.verticalScrollBar()->setValue(MyTextEdit.verticalScrollBar()->maximum());


    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput(), encoding='utf-8')
        #print (text) 
        self.append(text)

    def stderrReady(self):
        text = str(self.process.readAllStandardError())
        #print (text) #.strip())
        self.append(text)


#def main():
#    import sys
#    app = QApplication(sys.argv)
#    win = MainWindow()
#    win.show()
#    sys.exit(app.exec_())
    
#if __name__ == '__main__':
   # main()
