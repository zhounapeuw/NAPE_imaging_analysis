#!/usr/bin/env python
# coding: utf-8

# # NAPE Calcium Imaging Preprocessing Pipeline
#
# Note: differences between ipynb version: removed %%time in last cell
#
# Finds any .tif, .tiff, .h5 files in the requested directory and performs SIMA-based motion correction and fft-based bidirection 
# offset correction, signal extraction, and neuropil correction. This code parallelizes the computation at the session level by passing the multiple file paths (if there are more than one recordings) to the multiprocessing map function. 
# 
# __IMPORTANT RECOMENDATION__: This pipeline requires the user to manually draw regions-of-interest (ROIs) on the mean image (usually the motion-corrected output). __The ROI zip file must end in "_RoiSet" with the extension ".zip"__. If ROIs have not been drawn, it is recommended to use option 2 below (using files_to_analyze.py) and perform the preprocessing in two runs/executions of this code (main_parallel). For the __first run__, perform only the motion correction step. Take the H5 motion-corrected output and load it into FIJI (https://imagej.net/Fiji), manually draw ROIs, and save the ROIs. Then (__second run__) edit files_to_analyze.py now setting signal_extraction and neuropil_correction to True, and rerun this notebook/script (main_parallel).
# 
# How to run this code
# ------------------------------------
# 
# __In this jupyter notebook, just run all cells in order (shift + enter). When you reach the last cell, it will prompt the user for input. You have two options:__
# 
# 1) __Input the path to the root directory__ that contains the raw files. For example, if your files are in a folder called analyze_sessions: C:\Users\my_user\analyze_sessions  
# This will by default attempt to run motion correction, signal extraction, and neuropil extraction. You will encounter an error if ROI masks are not saved to the same directory as the raw data.
# 
# 2) You can also indicate specific files, parameters, and processing steps to include by __editing the python script called files_to_analyze.py__ (in the same directory as this main_parallel.ipynb). Once you have specified the files in files_to_analyze.py and saved, run this notebooks' cells, leave the input blank, and press enter; this code will automatically load the information in files_to_analyze.py.
# 
# To execute this in command line and follow the same directions as above:  
# `python main_parallel.py`
# 
# 
# See these documentations for details about SIMA
# ------------------------------------
# 
# https://github.com/losonczylab/sima  
# http://www.losonczylab.org/sima/1.3.2/  
# https://www.frontiersin.org/articles/10.3389/fninf.2014.00080/full
# 
# Required Packages
# -----------------
# Python 2.7, sima, glob, multiprocessing, numpy, h5py, pickle (optional if want to save displacement file) 
# 
# Custom code requirements: sima_motion_correction, bidi_offset_correction, calculate_neuropil (written by Vijay Namboodiri), files_to_analyze
# 
# Parameters (Only relevant if using the subfunction batch_process; ignore if using files_to_analyze or using default params by inputting a file directory)
# ----------
# 
# fdir : string
#     root file directory containing the raw tif, tiff, h5 files. Note: leave off the last backslash. For example: C:\Users\my_user\analyze_sessions
# 
# Optional Parameters (Only relevant if using batch_process)
# -------------------
# 
# max_disp : list of two entries
#     Each entry is an int. First entry is the y-axis maximum allowed displacement, second is the x-axis max allowed displacement.
#     The number of pixel shift for each line cannot go above these values.
#     Note: 50 pixels is approximately 10% of the FOV (512x512 pixels)
#     
#     Defaults to [30, 50]
#     
# save_displacement : bool 
#     Whether or not to have SIMA save the calculated displacements over time. def: False; NOTE: if this is switched to True,
#     it can double the time to perform motion correction.
#     
#     Defaults to False
#     
# Output
# -------
# motion corrected file (in the format of h5) with "\_sima_mc" appended to the end of the file name
# 
# "\*\_sima_masks.npy" : numpy data file  
#   * 3D array containing 2D masks for each ROI
# 
# "\*_extractedsignals.npy" : numpy data file  
#   * array containing pixel-averaged activity time-series for each ROI
#    
# "\_spatial_weights_*.h5" : h5 file  
#   * contains spatial weighting masks of neuropil for each ROI
# 
# "\_neuropil_signals_*.npy" : numpy data file  
#   * array containing neuropil signals for each ROI
# 
# "\_neuropil_corrected_signals_*.npy" : numpy data file  
#   * array containing neuropil-corrected signals for each ROI
# 
# "\*.json" : json file
#   * file containing the analysis parameters (fparams). Set by files_to_analyze.py or default parameters.
#   * to view the data, one can easily open in a text editor (eg. word or wordpad).
# 
# output_images : folder containing images  
#     You will also find a folder containing plots that reflect how each executed preprocessing step performed. Examples are mean images for motion corrected data, ROI masks overlaid on mean images, extracted signals for each ROI, etc..
# 
# note: * is a wildcard indicating additional characters present in the file name

# In[ ]:


# import native python packages
from fnmatch import fnmatch
import multiprocessing as mp
import os
from datetime import datetime

# import custom codes
import single_file_process
import files_to_analyze


# In[ ]:

def prep_fparams(root_dir, max_disp=[15, 15], save_displacement=False):

    if not root_dir:  # if string is empty, load predefined list of files in files_to_analyze

        fparams = files_to_analyze.define_fparams()

    else:

        root_dir = root_dir + '\\'

        # declare initialize variables to do with finding files to analyze
        fparams = []
        types = ['*.tif', '*.tiff', '*.h5']
        exclude_strs = ['spatialweights', '_sima_mc', '_trim_dims', '_offset_vals']

        # find files to analyze
        for path, subdirs, files in os.walk(root_dir):  # os.walk grabs all paths and files in subdirectories
            for name in files:
                # make sure file of any image file
                if any([fnmatch(name, ext) for ext in types]) and not any(
                        [exclude_str in name for exclude_str in exclude_strs]):  # but don't include processed files
                    tmp_dict = {}
                    tmp_dict['fname'] = name
                    tmp_dict['fdir'] = path
                    tmp_dict['max_disp'] = max_disp
                    tmp_dict['save_displacement'] = save_displacement

                    print(tmp_dict['fname'])
                    fparams.append(tmp_dict)

    return fparams

def batch_process(fparams):
    

    # print info to console
    num_files = len(fparams)
    if num_files == 0:
        raise Exception("No files to analyze!")
    print(str(num_files) + ' files to analyze')
    
    # determine number of cores to use and initialize parallel pool
    num_processes = min(mp.cpu_count(), num_files)
    print('Total CPU cores for parallel processing: ' + str(num_processes))
    pool = mp.Pool(processes=num_processes)
    
    # perform parallel processing; pass iterable list of file params to the analysis module selection code
    #pool.map(single_file_process.process, fparams)
    
    ## for testing
    for fparam in fparams:
        single_file_process.process(fparam)

    pool.close()
    pool.join()

    print('All done!')

if __name__ == "__main__":

    import matplotlib
    matplotlib.use('Agg')  # need to specify this line for cluster/server computing to make plotting non-interactive

    fdir = raw_input(r"Input root directory of tif, tiff, h5 files to analyze; note: Use FORWARD SLASHES to separate folder and leave the last backlash off!!  Otherwise leave blank to use files declared in file_to_analyze.py")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    fparams = prep_fparams(fdir)
    batch_process(fparams)


