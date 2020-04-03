# -*- coding: utf-8 -*-

import os
import sima_motion_bidi_correction
import sima_extract_roi_sig
import calculate_neuropil


def unpack(args):
    return process(*args)


def check_exist_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def process(fparams):
    fdir = fparams['fdir']
    fpath = os.path.join(fdir, fparams['fname']) # note fname contains file extension

    max_disp = fparams['max_disp']
    save_displacement = fparams['save_displacement']

    if not "motion_correct" in fparams:
        fparams['motion_correct'] = True
    if not "signal_extract" in fparams:
        fparams['signal_extract'] = True
    if not "npil_correct" in fparams:
        fparams['npil_correct'] = True

    # run motion correction
    if fparams['motion_correct']:
        sima_motion_bidi_correction.full_process(fpath, max_disp, save_displacement)
    # perform signal extraction
    if fparams['signal_extract']:
        sima_extract_roi_sig.extract(fpath)
    if fparams['npil_correct']:

        # make mean img output directory if it doesn't exist
        img_save_dir = check_exist_dir(os.path.join(fdir, os.path.splitext(fparams['fname'])[0] + '_output_images'))
        # make neuropil weight output plot directory if it doesn't exist
        npil_weight_save_dir = check_exist_dir(os.path.join(img_save_dir, 'npil_weights'))
        # make signal plot output directory if it doesn't exist
        signal_save_dir = check_exist_dir(os.path.join(img_save_dir, 'corr_signal'))

        # perform full neuropil correction
        calculate_neuropil.calculate_neuropil_signals_for_session(fpath)

        # plot and save figures from neuropil correction
        analyzed_data = calculate_neuropil.load_analyzed_data(fdir)
        calculate_neuropil.plot_ROI_masks(img_save_dir, analyzed_data['mean_img'],
                                          analyzed_data['masks'])
        calculate_neuropil.plot_deadzones(img_save_dir, analyzed_data['mean_img'],
                                          analyzed_data['h5weights']['deadzones_aroundrois'])
        calculate_neuropil.plot_npil_weights(npil_weight_save_dir, analyzed_data['mean_img'],
                                             analyzed_data['h5weights']['spatialweights'])
        calculate_neuropil.plot_corrected_sigs(signal_save_dir, analyzed_data['extract_signals'],
                                               analyzed_data['npil_corr_sig'], analyzed_data['npil_sig'])