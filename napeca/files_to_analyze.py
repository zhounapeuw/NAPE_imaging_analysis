# -*- coding: utf-8 -*-

"""

This script does not run any analysis/computations;
rather, here the user will specify each dataset to be analyzed along with
meta information, analysis parameters, and flags that indicate which processing steps to apply

User-defined input
------------------
Use the pre-written template in between the edit lines comment. To add additional files to analyze,
copy and paste the curly brackets (include the brackets!) and the contents within. Add a comma after
previous file's end curly bracket (if not there), and edit the dictionary values in the pasted text.

Here are descriptions of the editable values:

fname : string
    Raw data file name (can be h5, tif, tiff). Include the extension! Take care of if it is a .tif or .tiff!

fdir : string
    Path to the root directory of the raw file. I recommend using a raw string preceeded by the letter r to define a full path; see alternative example in the code below.
    NOTE: It is crucial to have the r in front of the string - this will make it a raw string and
        interpret the backslashes as such
    NOTE: there is no need for a last backslash

motion_correct: boolean
    Set to True if SIMA motion correction and bidirectional offset correction is desired; otherwise
    set to False

signal_extract: boolean
    Set to True if you want to perform average signal extraction from imageJ ROIs; otherwise set to False
    IMPORTANT: this step requires an "_mc.sima" folder (from sima motion correction) and imageJ ROI file(s)
        to run

npil_correct: boolean
    Set to True if you want to calculate and correct for each ROI's neuropil signal; Otherwise set to False.
    IMPORTANT: this step requires signal_extract to have been run or it set to True.

Optional Arguments
------------------

max_disp : list of two entries
    Each entry is an int. First entry is the y-axis maximum allowed displacement, second is the x-axis max allowed displacement.
    The number of pixel shift for each line cannot go above these values.
    Note: 50 pixels is approximately 10% of the FOV (512x512 pixels)

    Defaults to [30, 50]

flag_bidi_corr : boolean
    Whether or not to perform bidirection offset correction (cross-correlation of even/odd rows made by CZ)

    Defaults to True

save_displacement : boolean
    Whether or not to have SIMA save the calculated displacements over time. def: False; NOTE: if this is switched to True,
    it can double the time to perform motion correction.

    Defaults to False

fs : int or float
    Sampling rate of the input data

neuropil_radius : int
    This is the radius (in pixels) of the Gaussian weights for neuropil calculation (ie. the larger this value, the more
    of the surrounding space around the ROI will be included in the neuropil estimation
    Default will be 50 pixels

min_neuropil_radius : int
    This is the radius (in pixels) of a deadzone around an ROI that does not get counted towards the neuropil of any ROI.
    Note that this deadzone is applied to every ROI while calculating the neuropil for any ROI, not just the one
    whose neuropil is being calculated.
    Default will be 15 pixels

Output
------

fparams : list of dictionaries

    This is the iterable list of file information and parameters that gets passed to
    the preprocessing pipeline parallelizer function. Each list entry corresponds to a specific
    session.

For the fname, you will have to add the file extension to the string


"""

import os


def define_fparams():
    fparams = [

        # ONLY EDIT LINES BELOW THIS COMMENT

        {
            'fname': 'VJ_OFCVTA_7_260_D6_offset.h5',
            'fdir': os.path.abspath("../sample_data/VJ_OFCVTA_7_260_D6_offset"),
            # 'fdir': r'C:\Users\stuberadmin\Documents\GitHub\NAPE_imaging_analysis\sample_data\VJ_OFCVTA_7_260_D6_offset', # comment this in if want to use different absolute path
            'motion_correct': True,
            'signal_extract': False,
            'npil_correct': False,

            # optional args (can delete and will set to default values
            'max_disp': [10, 10],
            'neuropil_radius': 50,
            'min_neuropil_radius': 15,
            'fs': 5,

            'flag_save_displacement': False,
            'flag_bidi_corr': True,
            'flag_save_h5': True,
            'flag_save_projections': True,
        },

        {
            'fname': 'VJ_OFCVTA_8_300_D13_offset.h5',
            'fdir': os.path.abspath("../sample_data/VJ_OFCVTA_8_300_D13_offset"),
            'motion_correct': True,
            'signal_extract': False,
            'npil_correct': False,

            # optional args (can delete and will set to default values
            'max_disp': [10, 10],
            'neuropil_radius': 50,
            'min_neuropil_radius': 15,
            'fs': 5,

            'flag_save_displacement': False,
            'flag_bidi_corr': True,
            'flag_save_h5': True,
            'flag_save_projections': True,
        },

        # ONLY EDIT LINES ABOVE THIS COMMENT

    ]

    return fparams
