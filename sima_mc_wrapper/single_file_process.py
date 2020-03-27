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

        # make mean img output directory if it doesn't exist
        img_save_dir = os.path.join(fdir, 'output_images')
        if not os.path.exists(img_save_dir):
            os.mkdir(img_save_dir)

        # make neuropil weight output plot directory if it doesn't exist
        npil_weight_save_dir = os.path.join(img_save_dir, 'npil_weights')
        if not os.path.exists(npil_weight_save_dir):
            os.mkdir(npil_weight_save_dir)

        # make signal plot output directory if it doesn't exist
        signal_save_dir = os.path.join(img_save_dir, 'corr_signal')
        if not os.path.exists(signal_save_dir):
            os.mkdir(signal_save_dir)

        analyzed_data = calculate_neuropil.load_analyzed_data(fdir)

        calculate_neuropil.plot_ROI_masks(img_save_dir, analyzed_data['mean_img'],
                                          analyzed_data['masks'])
        calculate_neuropil.plot_deadzones(img_save_dir, analyzed_data['mean_img'],
                                          analyzed_data['h5weights']['deadzones_aroundrois'])
        calculate_neuropil.plot_npil_weights(npil_weight_save_dir, analyzed_data['mean_img'],
                                             analyzed_data['h5weights']['spatialweights'])
        calculate_neuropil.plot_corrected_sigs(signal_save_dir, analyzed_data['extract_signals'],
                                               analyzed_data['npil_corr_sig'], analyzed_data['npil_sig'])