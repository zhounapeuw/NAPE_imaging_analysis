import sima
import sima.motion
import numpy as np
import os
import pickle
import h5py
import sys
from sima import sequence
import bidi_offset_correction
from contextlib import contextmanager
import matplotlib
import matplotlib.pyplot as plt
import tifffile as tiff
import utils

# important for text to be detecting when importing saved figures into illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def unpack(args):
    print(args)
    return sima_motion_correction(*args)

# function that takes in mean image and plots
def subplot_mean_img(axs, data_name, mean_img, clims, zoom_window=None):
    im = axs.imshow(mean_img, cmap='gray')
    axs.set_title(data_name, fontsize=20)

    im.set_clim(vmin=clims[0], vmax=clims[1])

    if zoom_window is not None:
        axs.set_title(data_name + ' Zoom', fontsize=20)
        axs.axis(zoom_window)
        axs.invert_yaxis()
    axs.axis('off')

def save_mean_imgs(save_dir, data_raw, data_mc):

    # make image save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # compute mean images
    raw_mean = np.mean(np.squeeze(data_raw), axis=0)
    mc_mean = np.mean(np.squeeze(data_mc), axis=0)

    # calculate min and max array values across datasets to make color limits consistent
    clims = [np.min([np.min(raw_mean), np.min(mc_mean)]),
             np.max([np.max(raw_mean), np.max(mc_mean)])]
    print(list(clims))

    # make plot and save
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    subplot_mean_img(axs[0], 'Raw', raw_mean, clims)
    subplot_mean_img(axs[1], "Motion-Corrected", mc_mean, clims)
    plt.savefig(os.path.join(save_dir, 'raw_mc_imgs.png'))
    plt.savefig(os.path.join(save_dir, 'raw_mc_imgs.pdf'))


def save_projections(save_dir, data_in):

    max_img = utils.uint8_arr(np.max(data_in, axis=0))
    mean_img = utils.uint8_arr(np.mean(data_in, axis=0))
    std_img = utils.uint8_arr(np.std(data_in, axis=0))

    tiff.imwrite(os.path.join(save_dir, 'mean_img.tif'), mean_img)
    tiff.imwrite(os.path.join(save_dir, 'max_img.tif'), max_img)
    tiff.imwrite(os.path.join(save_dir, 'std_img.tif'), std_img)


def full_process(fpath, max_disp, save_displacement=False):

    """

    important note: sima saves a folder (.sima) that contains a sequences pickle file. This file contains the offsets
    calculated from the motion correction algorithm. Sima by itself does not save a new video/tiff dataset that is motion
    corrected.

    :param fpath:
    :param max_disp:
    :param save_displacement:
    :return:
    """

    print('Performing SIMA motion correction')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    fdir  = os.path.split(fpath)[0]
    fname = os.path.splitext(os.path.split(fpath)[1])[0]
    fext  = os.path.splitext(os.path.split(fpath)[1])[1]
    save_dir = os.path.join(fdir, fname + '_output_images')

    if fext == '.tif' or fext == '.tiff':
        # sequence: object that contains record of whole dataset; data not stored into memory all at once
        sequences = [sima.Sequence.create('TIFF', fpath)]
    elif fext == '.h5':
        sequences = [sima.Sequence.create('HDF5', fpath, 'tyx')]
    else:
        raise Exception('Inappropriate file extension')

    if not os.path.exists(os.path.join(fdir, fname + '_mc.sima')):

        # define motion correction method
        # n_processes can only handle =1! Bug in their code where >1 runs into an error
        # max_displacement: The maximum allowed displacement magnitudes in pixels in [y,x]
        mc_approach = sima.motion.HiddenMarkov2D(granularity='row', max_displacement=max_disp, n_processes=1, verbose=True)

        # apply motion correction to data
        dataset = mc_approach.correct(sequences, os.path.join(fdir, fname + '_mc.sima'),
                                      channel_names=['GCaMP'])
        # dataset dimensions are frame, plane, row(y), column (x), channel

        # use sima's fill_gaps function to interpolate missing data from motion correction
        # dtype can be changed to int16 since none of values are floats
        data_mc = np.empty(dataset[0]._sequences[0].shape, dtype='int16')
        filled_data = sequence._fill_gaps(iter(dataset[0]._sequences[0]), iter(dataset[0]._sequences[0]))
        for f_idx, frame in enumerate(filled_data):
            data_mc[f_idx, ...] = frame
        data_mc = np.squeeze(data_mc)

        if save_displacement is True:
            # show motion displacements after motion correction
            mcDisp_approach = sima.motion.HiddenMarkov2D(granularity='row', max_displacement=max_disp, n_processes=1,
                                                         verbose=True)
            displacements = mcDisp_approach.estimate(dataset)

            # save the resulting displacement file
            # only useful if you want to see the values of displacement calculated by SIMA to perform the motion correction
            displacement_file = open(os.path.join(fdir, fname + '_mc.sima/displacement.pkl'), "wb")
            pickle.dump(displacements, displacement_file)
            displacement_file.close()

            # process and save np array of composite displacement
            data_dims = displacements[0].shape
            disp_np = np.squeeze(np.array(displacements[0]))
            disp_meanpix = np.mean(disp_np, axis=1)  # avg across lines (y axis)

            sima_disp = np.sqrt(
                np.square(disp_meanpix[:, 0]) + np.square(disp_meanpix[:, 1]))  # calculate composite x + y offsets
            np.save(os.path.join(fdir, 'displacements\\displacements_sima.npy'), sima_disp)

        # perform bidirection offset correction
        my_bidi_corr_obj = bidi_offset_correction.bidi_offset_correction(data_mc)  # initialize data to object
        my_bidi_corr_obj.compute_mean_image()  # compute mean image across time
        my_bidi_corr_obj.determine_bidi_offset()  # calculated bidirectional offset via fft cross-correlation
        data_corrected, bidi_offset = my_bidi_corr_obj.correct_bidi_frames()  # apply bidi offset to data

        # save motion-corrected, bidi offset corrected dataset
        sima_mc_bidi_outpath = os.path.join(fdir, fname + '_sima_mc.h5')
        h5_write_bidi_corr = h5py.File(sima_mc_bidi_outpath, 'w')
        h5_write_bidi_corr.create_dataset('imaging', data=data_corrected)
        h5_write_bidi_corr.close()

        # save raw and mean images as figure
        save_mean_imgs(save_dir, np.array(sequences), data_corrected)
        # calculate and save projection images
        save_projections(save_dir, data_corrected)

        # sima by itself doesn't perform bidi corrections, so do so here:
        sequence_file = os.path.join(fdir, fname + '_mc.sima/sequences.pkl')
        sequence_data = pickle.load(open(sequence_file, "rb"))  # load the saved sequences pickle file
        sequence_data[0]['base']['displacements'][:, 0, 1::2, 1] += bidi_offset  # add bidi shift to existing offset values
        with open(sequence_file, 'wb') as handle:
            pickle.dump(sequence_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


