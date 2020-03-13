#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""

Finds any .tif, .tiff, .h5 files in the requested directory and performs SIMA-based motion correction and fft-based bidirection 
offset correction. This code parallelizes the computation at the session level by passing the multiple file paths (if there are 
more than one recordings) to the multiprocessing map function. The script sima_motion_correction contains the wrapping
code for SIMA and the custom-created class for bidirection offset correction.

Two simple ways to execute this in command line:  
A) sima_motion_correct_batch; then input the path_to_directory

B) sima_motion_correct_batch.batch_process(path_to_directory)

See these documentations for details
------------------------------------

https://github.com/losonczylab/sima
http://www.losonczylab.org/sima/1.3.2/
https://www.frontiersin.org/articles/10.3389/fninf.2014.00080/full

Required Packages
-----------------
sima, glob, multiprocessing, numpy, h5py, pickle (optional if want to save displacement file) 

Custom code requirements: sima_motion_correction, bidi_offset_correction

Parameters
----------

fdir : string
    root file directory containing the raw tif, tiff, h5 files 

Optional Parameters
-------------------

max_disp : list of two entries
    Each entry is an int. First entry is the y-axis maximum allowed displacement, second is the x-axis max allowed displacement.
    The number of pixel shift for each line cannot go above these values.
    Note: 50 pixels is approximately 10% of the FOV (512x512 pixels)
    
    Defaults to [30, 50]
    
save_displacement : bool 
    Whether or not to have SIMA save the calculated displacements over time. def: False; NOTE: if this is switched to True,
    it can double the time to perform motion correction.
    
    Defaults to False
    
Output
-------
motion corrected file (in the format of h5) with "_sima_mc" appended to the end of the file name
    

""";


# In[ ]:


import sima_motion_bidi_correction # reload(sima_motion_bidi_correction)
import glob
import multiprocessing as mp
import os


# In[ ]:


def batch_process(fdir, max_disp = [30, 50], save_displacement = False):

    # gather all tif or h5 files in root directory
    fpaths = []
    types = ['*.tif', '*.tiff', '*.h5']
    for type in types:
        fpaths.extend(glob.glob(fdir + type))
    print(str(len(fpaths)) + ' files to analyze')

    if len(fpaths) == 0:
        print("No files to analyze!")
    
    # determine number of cores to use and initialize parallel pool
    num_processes = min(mp.cpu_count(), len(fpaths))
    print( 'Total CPU cores for parallel processing: ' + str(num_processes) )
    pool = mp.Pool(processes = num_processes)
    
    # perform parallel processing; pass iterable list of file paths to the motion correction script
    pool.map(sima_motion_bidi_correction.unpack, [(file, max_disp) for file in fpaths])
    pool.close()
    pool.join()


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'if __name__ == "__main__":\n    fdir = raw_input("Input root directory of tif, tiff, h5 files to analyze; note: leave the last backlash off")\n    batch_process(fdir + \'\\\\\')')

