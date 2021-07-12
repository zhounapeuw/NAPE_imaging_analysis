
from PyQt4.QtGui import *
from PyQt4.uic import loadUi

from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

import numpy as np
import random


"""
mplwidget.py contains the class that defines the matplotlib widget plus canvas to draw on
"""

class MatplotlibWidget(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)

        loadUi("test_pyqt4_plot_img.ui", self)

        self.setWindowTitle("PyQt4 & Matplotlib Example GUI")

        self.pushButton_generate_random_signal.clicked.connect(self.update_graph)

        self.addToolBar(NavigationToolbar(self.graphics_1.canvas, self))

        self.addToolBar(NavigationToolbar(self.graphics_2.canvas, self))

    def update_graph(self):
        fs = 500
        f = random.randint(1, 100)
        ts = 1 / fs
        length_of_signal = 100
        t = np.linspace(0, 1, length_of_signal)

        cosinus_signal = np.cos(2 * np.pi * f * t)
        sinus_signal = np.sin(2 * np.pi * f * t)

        self.graphics_1.canvas.axes.clear()
        self.graphics_1.canvas.axes.plot(t, cosinus_signal)
        self.graphics_1.canvas.axes.plot(t, sinus_signal)
        self.graphics_1.canvas.axes.legend(('cosinus', 'sinus'), loc='upper right')
        self.graphics_1.canvas.axes.set_title('Cosinus - Sinus Signal')
        self.graphics_1.canvas.draw()

        fs = 500
        f = random.randint(1, 100)
        ts = 1 / fs
        length_of_signal = 100
        t = np.linspace(0, 1, length_of_signal)

        cosinus_signal = np.cos(2 * np.pi * f * t)
        sinus_signal = np.sin(2 * np.pi * f * t)

        self.graphics_2.canvas.axes.clear()
        self.graphics_2.canvas.axes.plot(t, cosinus_signal)
        self.graphics_2.canvas.axes.plot(t, sinus_signal)
        self.graphics_2.canvas.axes.legend(('cosinus', 'sinus'), loc='upper right')
        self.graphics_2.canvas.axes.set_title('Cosinus - Sinus Signal')
        self.graphics_2.canvas.draw()


app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()