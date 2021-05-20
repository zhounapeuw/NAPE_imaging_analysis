import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5.uic import loadUi
import numpy as np
import os

import main_parallel

class MainWindow(QMainWindow):

    def __init__(self):

        super(MainWindow, self).__init__()
        loadUi(r"C:\Users\stuberadmin\Documents\GitHub\NAPE_imaging_analysis\napeca\qt_design_gui.ui", self)

        self.button_browse_files.clicked.connect(self.getfiles)
        self.button_start_preprocess.clicked.connect(self.start_preprocess)

        # setup table object
        self.table_fparams = self.findChild(QtWidgets.QTableView, 'table_fparams')
        # associate table with model
        self.model_fparam_table = QtGui.QStandardItemModel(4, 3, self)
        self.table_fparams.setModel(self.model_fparam_table)

        self.populateTable()

        self.setWindowTitle("UI Testing")

    def populateTable(self, fpaths=[]):
        """

        takes in a list of files paths, extracts file directory and name, then puts info into qTable

        :param fpaths:
        :return:
        """
        self.fparam_order = ['fname', 'fdir', 'max_disp_y',
                        'max_disp_x']  # internal usage: edit this if adding more parameters

        self.model_fparam_table.clear()  # reset table data
        self.model_fparam_table.setHorizontalHeaderLabels(self.fparam_order)  # set column header names

        fparams = []

        default_param_values = {'max_disp_y': '~', 'max_disp_x': '~'}

        if not fpaths:  # populate with sample data when first starting app
            fpaths = [os.path.abspath("../sample_data/VJ_OFCVTA_8_300_D13_offset/VJ_OFCVTA_8_300_D13_offset.tif"),
                      os.path.abspath("../sample_data/VJ_OFCVTA_7_260_D6_offset/VJ_OFCVTA_7_260_D6_offset.h5")]

        # create a list of dicts that contains each file's parameters
        for file_idx, fpath in enumerate(fpaths):

            file_tmp_dict = {}  # using a dict to make it easier to pull values for certain parameters into the table

            # internal usage: add to below if adding more parameters
            file_tmp_dict['fname'] = os.path.splitext(os.path.basename(fpath))[0]
            file_tmp_dict['fdir'] = os.path.dirname(fpath)
            file_tmp_dict['fext'] = os.path.splitext(os.path.basename(fpath))[1]
            file_tmp_dict['max_disp_y'] = '15'  # CZ placeholder
            file_tmp_dict['max_disp_x'] = '15'

            fparams.append(file_tmp_dict)

        # populate table
        for row_idx, file_fparam in enumerate(fparams):
            for col_idx, param_name in enumerate(self.fparam_order):
                item = QtGui.QStandardItem(file_fparam[param_name])
                self.model_fparam_table.setItem(row_idx, col_idx, item)


    def getfiles(self):
        dlg = QFileDialog(self, 'Select h5 or tif of recording',
                          r'C:\Users\stuberadmin\Documents\GitHub\NAPE_imaging_analysis\sample_data\VJ_OFCVTA_7_260_D6_offset')
        dlg.setFileMode(QFileDialog.ExistingFiles) # allow for multiple files to be selected
        dlg.setNameFilters(["Images (*.h5 *.tif *.tiff)"]) # filter for specific ftypes

        if dlg.exec_():
            fpaths = dlg.selectedFiles()

        self.populateTable(fpaths)

        return fpaths

    def start_preprocess(self):
        """
        Converts fparams table in gui to list of dictionaries for main_parallel.py

        :return:
        """
        fparams = []

        for row in range(self.model_fparam_table.rowCount()):
            file_tmp_dict = {}
            for col in range(self.model_fparam_table.columnCount()):

                column_name = self.fparam_order[col]

                # internal usage: add to below if adding more parameters
                file_tmp_dict[column_name] = self.model_fparam_table.item(row, col).text()
                if column_name in ['max_disp_y', 'max_disp_x']:
                    file_tmp_dict[column_name] = int(self.model_fparam_table.item(row, col).text())


            fparams.append(file_tmp_dict)

        #main_parallel.batch_process(fparams)

app = QApplication(sys.argv)

mainwindow=MainWindow()
widget=QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.resize(1200, 800)

widget.show()

sys.exit(app.exec_())