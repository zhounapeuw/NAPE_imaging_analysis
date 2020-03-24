# -*- coding: utf-8 -*-

import os
import sima_motion_bidi_correction
import sima_extract_roi_sig
import calculate_neuropil

def unpack(args):
    return process(*args)

def process(fparams):
    mc_flag = False; sig_extr_flag = True; npil_flag = True

    fdir = fparams['fdir']
    fpath = os.path.join(fdir, fparams['fname'])
    max_disp = fparams['max_disp']
    save_displacement = fparams['save_displacement']

    if mc_flag:
        sima_motion_bidi_correction.full_process(fpath, max_disp, save_displacement)

    if sig_extr_flag:
        sima_extract_roi_sig.extract(fpath)

    if npil_flag:
        calculate_neuropil.calculate_neuropil_signals_for_session(fdir)