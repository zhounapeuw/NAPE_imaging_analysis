# -*- coding: utf-8 -*-

import os
import sima_motion_bidi_correction
import sima_extract_roi_sig
import calculate_neuropil


def unpack(args):
    return process(*args)


def process(fparams):

    fdir = fparams['fdir']
    fpath = os.path.join(fdir, fparams['fname'])

    max_disp = fparams['max_disp']
    save_displacement = fparams['save_displacement']

    if not "motion_correct" in fparams:
        fparams['motion_correct'] = True
    if not "signal_extract" in fparams:
        fparams['signal_extract'] = True
    if not "npil_correct" in fparams:
        fparams['npil_correct'] = True

    if fparams['motion_correct']:
        sima_motion_bidi_correction.full_process(fpath, max_disp, save_displacement)
    if fparams['signal_extract']:
        sima_extract_roi_sig.extract(fpath)
    if fparams['npil_correct']:
        calculate_neuropil.calculate_neuropil_signals_for_session(fdir)
        calculate_neuropil.plot_ROI_masks(fdir)