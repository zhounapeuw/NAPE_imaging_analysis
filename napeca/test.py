from PyQt4 import QtCore, QtGui
import numpy as np

class Widget(QtGui.QWidget):

    def __init__(self, *args, **kwargs):
        super(Widget, self).__init__(*args, **kwargs)
        self.resize(800,600)

        self.vlayout = QtGui.QVBoxLayout(self)
        self.table = QtGui.QTableView()
        self.vlayout.addWidget(self.table)

        self.hlayout = QtGui.QHBoxLayout()
        self.list1 = QtGui.QListView()
        self.list2 = QtGui.QListView()
        self.list3 = QtGui.QListView()
        self.hlayout.addWidget(self.list1)
        self.hlayout.addWidget(self.list2)
        self.hlayout.addWidget(self.list3)

        self.vlayout.addLayout(self.hlayout)

        # setup table object
        self.model = QtGui.QStandardItemModel(4,3,self)
        self.table.setModel(self.model)
        self.model.setHorizontalHeaderLabels(['File name', 'File dir', 'Max Disp'])

        # initialize lists
        self.list1.setModel(self.model)
        self.list1.setModelColumn(0)
        self.list2.setModel(self.model)
        self.list2.setModelColumn(1)
        self.list3.setModel(self.model)
        self.list3.setModelColumn(2)

        self.populateTable()

        self.print_table_values()

    def populateTable(self):
        data = np.array([['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9'], ['10', '11', '12']])
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                item = QtGui.QStandardItem(data[row, col])
                self.model.setItem(row, col, item)

    def print_table_values(self):

        nb_row = 4
        nb_col = 3

        for row in range(nb_row):
            for col in range(nb_col):
                print(str(self.model.item(row, col).text()))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication([])
    window = Widget()
    window.show()
    window.raise_()
    sys.exit(app.exec_())