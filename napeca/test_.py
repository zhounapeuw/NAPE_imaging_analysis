
from PyQt4.QtCore import *
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QApplication, QFileDialog, QMainWindow

import time
import sys


class Stream(QtCore.QObject):
    """Redirects console output to text widget."""
    newText = QtCore.pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

class MainWindow(QtGui.QDialog):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.threadpool = QThreadPool()

        button = QtGui.QPushButton(self)
        button.setText('Run and output text')
        button.pressed.connect(self.api)

        # Custom output stream.
        sys.stdout = Stream(newText=self.onUpdateText)

        # Create the text output widget.
        self.process = QtGui.QTextEdit(self, readOnly=True)
        self.process.ensureCursorVisible()
        self.process.setLineWrapColumnOrWidth(500)
        self.process.setLineWrapMode(QtGui.QTextEdit.FixedPixelWidth)
        self.process.setFixedWidth(400)
        self.process.setFixedHeight(150)
        self.process.move(30, 100)

        # Set window size and title, then show the window.
        self.setGeometry(300, 300, 600, 300)
        self.setWindowTitle('Generate Master')
        self.show()

        self.show()

    def onUpdateText(self, text):
        """Write console output to text widget."""
        cursor = self.process.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.process.setTextCursor(cursor)
        self.process.ensureCursorVisible()

    def api(self):
        worker = Worker(self.genMastClicked)
        self.threadpool.start(worker)

    def genMastClicked(self):
        """Runs the main function."""
        print('Running...')
        time.sleep(5)
        print('Done.')

class Worker(QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):

        self.fn(*self.args, **self.kwargs)


app = QApplication([])
window = MainWindow()
app.exec_()