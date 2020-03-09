#!/usr/bin/env python
# coding: utf-8

# In[ ]:



"""
Finds any .tif, .tiff, .h5 files in the requested directory and performs SIMA-based motion correction and fft-based bidirection 
offset correction. This code parallelizes the computation at the session level by passing the multiple file paths (if there are 
more than one recordings) to the multiprocessing map function. The script sima_motion_correction contains the wrapping
code for SIMA and the custom-created class for bidirection offset correction.

See these documentations for details:

https://github.com/losonczylab/sima
http://www.losonczylab.org/sima/1.3.2/
https://www.frontiersin.org/articles/10.3389/fninf.2014.00080/full

required packages: 
sima, glob, multiprocessing, numpy, h5py, pickle (optional if want to save displacement file) 

Parameters
----------

fdir : string
    root file directory. 

max_disp : list of two entries
    Each entry is an int. First entry is the y-axis maximum allowed displacement, second is the x-axis max allowed displacement.
    The shift for each line cannot go above these values.

save_displacement : bool 
    Whether or not to have SIMA save the calculated displacements over time. def: False; NOTE: if this is switched to True,
    it can double the time to perform motion correction.

Output
-------
motion corrected file (in the format of h5) with "_sima_mc" appended to the end of the file name
    

"""


# In[ ]:


import sima_motion_correction
reload(sima_motion_correction)
import glob
import multiprocessing as mp


# In[ ]:


def batch_process(fdir, max_disp = [30, 50], save_displacement = False):

    fdir = 'C:\\2pData\\Sean\\'
    max_disp = [30, 50] 
    save_displacement = False

    # gather all tif or h5 files in root directory
    fpaths = []
    types = ['*.tif', '*.tiff', '*.h5']
    for type in types:
        fpaths.extend(glob.glob(fdir + type))
    print(str(len(fpaths)) + ' files to analyze')

    # perform parallel processing
    num_processes = min(mp.cpu_count(), len(fpaths))
    print( 'Total CPU cores for parallel processing: ' + str(num_processes) )

    pool = mp.Pool(processes = num_processes)
    pool.map(sima_motion_correction.unpack, [(file, max_disp) for file in fpaths])
    pool.close()
    pool.join()


# In[ ]:


if __name__ == "__main__":
    fdir = raw_input()
    batch_process(fdir)


# In[ ]:




