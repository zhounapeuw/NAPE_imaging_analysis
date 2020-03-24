# -*- coding: utf-8 -*-

"""

For the fname, you will have to add the file extension to the string


"""


def define_fparams():

    fparams = [ {
                'fname'         : 'itp_lhganiii_p7ml_920_0001_tiff',
                'fdir'          : r'C:\2pData\Ivan\test\itp_lhganiii_p7ml_920_0001',
                'motion_correct': True,
                'signal_extract': True,
                'npil_correct'  : True,
                'max_disp': [30, 50],
                'save_displacement': False
                },

             {
                'fname'         : 'itp_lhganiv_quin_680_0001_tiff',
                'fdir'          : r'C:\2pData\Ivan\test\itp_lhganiv_quin_680_0001',
                'motion_correct': True,
                'signal_extract': True,
                'npil_correct'  : True,
                'max_disp': [30, 50],
                'save_displacement': False
                },

            ]

    return fparams
