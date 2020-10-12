# -*- coding: utf-8 -*-

import os
import sima_motion_bidi_correction
import sima_extract_roi_sig
import calculate_neuropil
import sima
import sys
import json
from datetime import datetime
import time


def unpack(args):
    return process(*args)


def check_exist_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def check_create_sima_dataset(fpath):
    fdir = os.path.split(fpath)[0]
    fname = os.path.split(fpath)[1]
    fbasename = os.path.splitext(fname)[0]
    fext = os.path.splitext(fname)[1]

    sima_folder_path = os.path.join(fdir, fbasename + '_mc.sima')

    if not os.path.exists(sima_folder_path):

        # create a sima sequence with the data
        if fext == '.tif' or fext == '.tiff':
            sequences = [sima.Sequence.create('TIFF', fpath)]
        elif fext == '.h5':
            sequences = [sima.Sequence.create('HDF5', fpath, 'tyx')]
        # creates sima imaging dataset, but more importantly saves a .sima folder required for downstream processing
        sima.ImagingDataset(sequences, sima_folder_path);

def save_json_dict(savedir, fname, dict_):
    savepath = os.path.join(savedir, fname + '.json')
    with open(savepath, 'w') as fp:
        json.dump(dict_, fp)


def set_fparam_defaults(fparams):
    if "motion_correct" not in fparams:
        fparams['motion_correct'] = True
    if "signal_extract" not in fparams:
        fparams['signal_extract'] = True
    if "npil_correct" not in fparams:
        fparams['npil_correct'] = True
    if "flag_bidi_corr" not in fparams:
        fparams['flag_bidi_corr'] = True
    return fparams


def process(fparams):

    fpath = os.path.join(fparams['fdir'], fparams['fname'])  # note fname contains file extension
    fbasename = os.path.splitext(fparams['fname'])[0]

    # set default parameters if variables not defined
    fparams = set_fparam_defaults(fparams)

    # run motion correction
    if fparams['motion_correct']:
        sima_motion_bidi_correction.full_process(fpath, fparams['max_disp'], fparams['flag_bidi_corr'], fparams['save_displacement'])
    else:
        check_create_sima_dataset(fpath)

    # perform signal extraction
    if fparams['signal_extract']:
        start_time = time.time()
        sima_extract_roi_sig.extract(fpath)
        end_time = time.time()
        print("Signal extraction execution time: {} seconds".format(end_time - start_time))

    # perform neuropil extraction and correction
    if fparams['npil_correct']:
        start_time = time.time()
        # make mean img output directory if it doesn't exist
        img_save_dir = check_exist_dir(os.path.join(fparams['fdir'], fbasename + '_output_images'))
        # make neuropil weight output plot directory if it doesn't exist
        npil_weight_save_dir = check_exist_dir(os.path.join(img_save_dir, 'npil_weights'))
        # make signal plot output directory if it doesn't exist
        signal_save_dir = check_exist_dir(os.path.join(img_save_dir, 'corr_signal'))

        # perform full neuropil correction
        calculate_neuropil.calculate_neuropil_signals_for_session(fpath, fparams)

        # plot and save figures from neuropil correction
        analyzed_data = calculate_neuropil.load_analyzed_data(fparams['fdir'], fparams['fname'])
        calculate_neuropil.plot_ROI_masks(img_save_dir, analyzed_data['mean_img'],
                                          analyzed_data['masks'])
        calculate_neuropil.plot_deadzones(img_save_dir, analyzed_data['mean_img'],
                                          analyzed_data['h5weights']['deadzones_aroundrois'])
        calculate_neuropil.plot_npil_weights(npil_weight_save_dir, analyzed_data['mean_img'],
                                             analyzed_data['h5weights']['spatialweights'])
        calculate_neuropil.plot_corrected_sigs(signal_save_dir, analyzed_data['extract_signals'],
                                               analyzed_data['npil_corr_sig'], analyzed_data['npil_sig'], fparams)

        end_time = time.time()
        print("Npil correction execution time: {} seconds".format(end_time - start_time))

    # datetime object containing current date and time
    fparams['date_time'] = str(datetime.now())
    save_json_dict(fparams['fdir'], fbasename, fparams)