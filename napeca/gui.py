import sys
import os
import glob
import h5py
import numpy as np
import tifffile as tiff
import sima
from functools import partial
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.uic import loadUi
# imports for interfacing matplotlib to pyqt
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

import main_parallel




# worker framework to output live stdout: https://stackoverflow.com/questions/50767240/flushing-output-directed-to-a-qtextedit-in-pyqt
class text_stream(QtCore.QObject):
    """Redirects console output to text widget."""
    newText = QtCore.pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))


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


class MainWindow(QMainWindow):

    def __init__(self):

        super(MainWindow, self).__init__()
        loadUi("./qt_design_gui.ui", self)

        self.threadpool = QThreadPool()
        if sys.gettrace() is None:
            # Custom output stream for live stdout in text box (outputs prints throughout the script)
            sys.stdout = text_stream(newText=self.onUpdateText)  # THIS MESSES WITH PYCHARM'S DEBUG MODE; COMMENT OUT FOR DEBUGGING

        # initialize graphics views for plotting projection images
        self.addToolBar(NavigationToolbar(self.graphics_1.canvas, self))
        self.addToolBar(NavigationToolbar(self.graphics_2.canvas, self))

        # initialize buttons
        self.button_browse_files.clicked.connect(self.getfiles)
        self.button_start_preprocess.clicked.connect(self.worker_start_preprocess)
        self.button_delete_row.clicked.connect(self.delete_row)
        self.button_duplicate_param.clicked.connect(self.duplicate_param)
        self.button_clear_table.clicked.connect(self.clear_fparam_table)
        self.button_plot_mean.clicked.connect(partial(self.update_graph, 'mean'))
        self.button_plot_max.clicked.connect(partial(self.update_graph, 'max'))
        self.button_plot_std.clicked.connect(partial(self.update_graph, 'std'))
        self.button_halp.clicked.connect(self.get_halp_window)

        # initialize text box for printed ouput
        self.text_box_output_obj = self.findChild(QtGui.QPlainTextEdit, 'text_box_output')
        self.text_box_output_obj.setReadOnly(True)

        # setup table object
        self.view_table_fparams = self.findChild(QtGui.QTableView, 'table_fparams')  # for pyqt5, replace QtGui with QWidget
        # associate table with model
        self.model_fparam_table = QtGui.QStandardItemModel(7, 2, self)
        self.view_table_fparams.setModel(self.model_fparam_table)

        self.populate_table()

        self.setWindowTitle("Preprocessing")


    ### methods for live stdout printing
    def onUpdateText(self, text):
        """Write console output to text widget."""
        cursor = self.text_box_output_obj.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.text_box_output_obj.setTextCursor(cursor)
        self.text_box_output_obj.ensureCursorVisible()

    def worker_start_preprocess(self):
        worker = Worker(self.start_preprocess)
        self.threadpool.start(worker)


    ### methods for initializing data for QTable
    def clear_fparam_table(self):

        self.model_fparam_table.clear()  # reset table data
        self.model_fparam_table.setHorizontalHeaderLabels(self.fparam_order)  # set column header names

    def initialize_default_fpaths(self):

        """
        Just creates fpaths based on sample data
        Child function and required for populate_table method

        :return:
        """

        fpaths = [os.path.abspath("../sample_data/VJ_OFCVTA_8_300_D13_offset/VJ_OFCVTA_8_300_D13_offset.tif"),
                  os.path.abspath("../sample_data/VJ_OFCVTA_7_260_D6_offset/VJ_OFCVTA_7_260_D6_offset.h5")]

        return fpaths

    def set_create_fparams_list(self, fpaths):

        """

        Takes in list of fpaths and turns it into a list of dicts (for populating into table)
        Child function and required for populate_table method

        :param fpaths:
        :return:
        """

        self.fparam_order = ['fname', 'fdir', 'fs', 'max_disp_y',
                             'max_disp_x', 'motion_correct', 'signal_extract',
                             'npil_correct', 'flag_save_projections', 'flag_save_h5']  # internal usage: edit this if adding more parameters

        # create a list of dicts that contains each file's parameters
        fparams = []
        for file_idx, fpath in enumerate(fpaths):
            file_tmp_dict = {}  # using a dict to make it easier to pull values for certain parameters into the table

            # internal usage: add to below if adding more parameters
            file_tmp_dict['fname'] = os.path.basename(fpath)
            file_tmp_dict['fdir'] = os.path.dirname(fpath)
            file_tmp_dict['fs'] = '5'
            file_tmp_dict['max_disp_y'] = '20'  # CZ placeholder
            file_tmp_dict['max_disp_x'] = '20'
            file_tmp_dict['motion_correct'] = 'True'
            file_tmp_dict['signal_extract'] = 'True'
            file_tmp_dict['npil_correct'] = 'True'
            file_tmp_dict['flag_save_projections'] = 'False'
            file_tmp_dict['flag_save_h5'] = 'False'

            fparams.append(file_tmp_dict)

        return fparams

    # make a dict of possible paths for loading and saving
    def define_paths(self, fdir, fname):
        self.paths_dict = {'fdir': fdir, 'fname': fname}
        self.paths_dict['sima_projection_folder'] = os.path.join(fdir, '{}_output_images'.format(fname))


    ### methods for QTable
    def populate_table(self, fpaths=[]):
        """

        takes in a list of files paths, extracts file directory and name, then puts info into qTable

        :param fpaths:
        :return:
        """

        if not fpaths:
            fpaths = self.initialize_default_fpaths()
            self.model_fparam_table.clear()  # reset table data

        self.num_recs = len(fpaths)

        fparams = self.set_create_fparams_list(fpaths)
        self.model_fparam_table.setHorizontalHeaderLabels(self.fparam_order)  # set column header names

        # populate table
        current_num_rows = self.model_fparam_table.rowCount()
        for row_idx, file_fparam in enumerate(fparams):
            for col_idx, param_name in enumerate(self.fparam_order):
                item = QtGui.QStandardItem(file_fparam[param_name])  # convert string to QStandardItem
                self.model_fparam_table.setItem(row_idx+current_num_rows, col_idx, item) # add item to table model

    def getfiles(self):
        dlg = QFileDialog(self, 'Select h5 or tif of recording',
                          os.path.abspath(r'../sample_data'))
        dlg.setFileMode(QFileDialog.ExistingFiles)  # allow for multiple files to be selected
        dlg.setNameFilters(["Images (*.h5 *.tif *.tiff)"])  # filter for specific ftypes

        if dlg.exec_():
            fpaths = dlg.selectedFiles()
            fpaths = [str(f) for f in fpaths]  # for pyqt4; turn QStringList to python list

        self.populate_table(fpaths)
        self.num_recs = self.model_fparam_table.rowCount()

    def delete_row(self):

        index_list = []
        for model_index in self.view_table_fparams.selectionModel().selectedRows():
            index = QtCore.QPersistentModelIndex(model_index)
            index_list.append(index)

        for index in index_list:
            self.model_fparam_table.removeRow(index.row())  # have to delete row from model, not the view (table_fparams)

    def _get_selected_table_val(self):
        """

        :return:

        output Var1: content of highlighted cell as a string
        output Var2: QModelIndex object, invoke methods row() and column() to get indices
        """
        index = self.view_table_fparams.selectionModel().currentIndex()
        return str(self.view_table_fparams.model().data(index).toString()), index

    def duplicate_param(self):

        value_to_duplicate, index_obj = self._get_selected_table_val()

        for row_idx in range(self.num_recs):
            item = QtGui.QStandardItem(value_to_duplicate)
            self.model_fparam_table.setItem(row_idx, index_obj.column(), item)

    ### methods for plotting images
    def load_frames(self, fdir, fname):

        fname_base, fext = os.path.splitext(fname)  # split fname for extension
        if fext == '.h5':
            h5 = h5py.File(os.path.join(fdir, fname), 'r')
            return h5[list(h5.keys())[0]]
        elif fext == '.tif':
            return tiff.imread(os.path.join(fdir, fname)).astype('uint16')

    def get_subset_frames(self, data):

        nframes_min = 300
        num_frames = data.shape[0]

        if num_frames > nframes_min:
            frames2avg = np.unique(np.linspace(0, num_frames, nframes_min).astype(int))
        else:
            frames2avg = np.arange(num_frames)
        return np.array(np.squeeze(data[frames2avg, ...])).astype('uint16')  # np.array loads all data into memory

    def make_projections(self, arr_in, proj_type):

        if proj_type == 'mean':
            return np.mean(arr_in, axis=0)
        elif proj_type == 'std':
            return np.std(arr_in, axis=0)
        elif proj_type == 'max':
            return np.max(arr_in, axis=0)

    def update_graph(self, proj_type='mean'):

        # grab highlighted row's projection images if available
        _, cell_index = self._get_selected_table_val()
        if cell_index > 0:
            fdir = str(self.model_fparam_table.item(cell_index.row(), self.fparam_order.index('fdir')).text())
            fname = str(self.model_fparam_table.item(cell_index.row(), self.fparam_order.index('fname')).text())

            # first load raw data, then get subset of frames, then make projection
            to_plot = self.make_projections(self.get_subset_frames(self.load_frames(fdir, fname)), proj_type)

            self.graphics_1.canvas.axes.clear()
            self.graphics_1.canvas.axes.imshow(to_plot)
            self.graphics_1.canvas.axes.set_title("{} \n Raw {}".format(fname, proj_type))
            self.graphics_1.canvas.draw()

            if os.path.exists(os.path.join(fdir, os.path.splitext(fname)[0] + '_mc.sima')):
                dataset = sima.ImagingDataset.load(os.path.join(fdir, os.path.splitext(fname)[0] + '_mc.sima'))
                to_plot = self.make_projections(self.get_subset_frames((np.squeeze(dataset[0]._sequences[0]))), proj_type)

                self.graphics_2.canvas.axes.clear()
                self.graphics_2.canvas.axes.imshow(to_plot)
                self.graphics_2.canvas.axes.set_title('Motion-Corrected {}'.format(proj_type))
                self.graphics_2.canvas.draw()

    def get_halp_window(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("""
        0) Two sample datasets are prepopulated in the data-to-analyze table. Click preprocess to start the analysis
        
        1) To analyze new data, click the "clear table" button and browse to a .tif or .h5 file using the "browse file" button.
        
        2) Once files have been specified, default parameters for analysis are loaded. You can change the parameters for each recording by double-clicking the cell of interest and typing in the desired value/entry. 
        
        fname: file basename
        fdir: file root directory
        fs: sampling rate of raw data
        max_disp_y: maximum expected motion shifts in y axis
        max_disp_x: maximum expected motion shifts in x axis
        
        3) Click the "Start Preprocess" button; information about the analysis will populate the dialog box below.
        
        4) After motion correction completes, you can click the plotting buttons to visualize the data.
        
        """)
        msg.setWindowTitle("Short how-to")
        msg.setDetailedText("More details:")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        __ = msg.exec_()

    def start_preprocess(self):
        """
        Converts fparams table in gui to list of dictionaries for main_parallel.py

        :return:
        """

        # needed b/c x and y are separate in gui table, but needs to be a single list for main_parallel analysis
        def combine_displacements(fparam_dict):
            fparam_dict['max_disp'] = [fparam_dict['max_disp_y'], fparam_dict['max_disp_x']]
            return fparam_dict

        fparams = []

        for row in range(self.model_fparam_table.rowCount()):
            file_tmp_dict = {}
            for col in range(self.model_fparam_table.columnCount()):

                column_name = self.fparam_order[col]

                # internal usage: add to below if adding more parameters
                file_tmp_dict[column_name] = self.model_fparam_table.item(row, col).text()
                if 'QString' in str(type(self.model_fparam_table.item(row, col).text())):  # PyQT4 needs text() output to be converted to string
                    file_tmp_dict[column_name] = str(file_tmp_dict[column_name])
                # not really another way around this - turn string containing boolean to boolean
                if file_tmp_dict[column_name].lower() in ['true', 't', 'y']:
                    file_tmp_dict[column_name] = True
                elif file_tmp_dict[column_name].lower() in ['false', 'f', 'n']:
                    file_tmp_dict[column_name] = False

                if column_name in ['max_disp_y', 'max_disp_x', 'fs']:
                    file_tmp_dict[column_name] = float(self.model_fparam_table.item(row, col).text())

            file_tmp_dict = combine_displacements(file_tmp_dict)

            fparams.append(file_tmp_dict)

        main_parallel.batch_process(fparams)

if __name__ == '__main__':

    app = QApplication(sys.argv)

    mainwindow = MainWindow()
    widget = QtGui.QStackedWidget()  # for pyqt5, replace QtGui with QWidget
    widget.addWidget(mainwindow)
    widget.resize(1200, 1000)
    widget.show()

    sys.exit(app.exec_())