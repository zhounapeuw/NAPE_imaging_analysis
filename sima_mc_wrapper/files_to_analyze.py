# -*- coding: utf-8 -*-

"""

For the fname, you will have to add the file extension to the string


"""

def define_fparams():
    fparams = [

        {
            'fname': 'VJ_OFCVTA_7_260_D6_offset.h5',
            'fdir': r'C:\Users\stuberadmin\Documents\GitHub\NAPE_imaging_analysis\sample_data\VJ_OFCVTA_7_260_D6_offset',
            'motion_correct': False,
            'signal_extract': True,
            'npil_correct': True,
            'max_disp': [30, 50],
            'save_displacement': False
        },

        {
            'fname': 'VJ_OFCVTA_7_260_D6.h5',
            'fdir': r'C:\Users\stuberadmin\Documents\GitHub\NAPE_imaging_analysis\sample_data\VJ_OFCVTA_7_260_D6',
            'motion_correct': False,
            'signal_extract': True,
            'npil_correct': True,
            'max_disp': [30, 50],
            'save_displacement': False
        }

    ]

    return fparams
