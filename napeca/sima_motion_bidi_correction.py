import sima
import sima.motion
import numpy as np
import os
import pickle
import h5py
import sys
from sima import sequence
import time
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

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    max_img = utils.uint8_arr(np.max(data_in, axis=0))
    mean_img = utils.uint8_arr(np.mean(data_in, axis=0))
    std_img = utils.uint8_arr(np.std(data_in, axis=0))

    tiff.imwrite(os.path.join(save_dir, 'mean_img.tif'), mean_img)
    tiff.imwrite(os.path.join(save_dir, 'max_img.tif'), max_img)
    tiff.imwrite(os.path.join(save_dir, 'std_img.tif'), std_img)


def save_projections_chunked(fdir, fname, save_dir):

    fpath = os.path.join(fdir, fname + '.h5')
    h5 = h5py.File(fpath, 'r')
    h5_obj = h5[list(h5.keys())[0]]

    num_frames = h5_obj.shape[0]
    chunk_size = 3000.0
    n_chunks = int(np.ceil(num_frames / chunk_size))
    chunked_frame_idx = np.array_split(np.arange(num_frames), n_chunks)  # split frame indices into chunks

    frame_mean_chunks = np.empty([n_chunks, h5_obj.shape[-2], h5_obj.shape[-1]])
    frame_std_chunks = np.empty([n_chunks, h5_obj.shape[-2], h5_obj.shape[-1]])
    frame_max_chunks = np.empty([n_chunks, h5_obj.shape[-2], h5_obj.shape[-1]])

    for chunk_idx, frame_idx in enumerate(chunked_frame_idx):
        print('projecting from frame {} to {}'.format(frame_idx[0], frame_idx[-1]))
        chunk_data = np.squeeze(np.array(h5_obj[frame_idx, ...])).astype('int16')  # np.array loads all data into memory
        frame_mean_chunks[chunk_idx, ...] = np.mean(chunk_data, axis=0)
        frame_std_chunks[chunk_idx, ...] = np.std(chunk_data, axis=0)
        frame_max_chunks[chunk_idx, ...] = np.max(chunk_data, axis=0)
    h5.close()

    all_frame_mean = np.squeeze(np.mean(frame_mean_chunks, axis=0))
    all_frame_std = np.squeeze(np.mean(frame_std_chunks, axis=0))
    all_frame_max = np.squeeze(np.mean(frame_max_chunks, axis=0))

    tiff.imwrite(os.path.join(save_dir, 'mean_img.tif'), all_frame_mean)
    tiff.imwrite(os.path.join(save_dir, 'max_img.tif'), all_frame_max)
    tiff.imwrite(os.path.join(save_dir, 'std_img.tif'), all_frame_std)

def apply_bidi_corr_to_sima_offsets(fdir, fname, bidi_offset):
    # sima by itself doesn't perform bidi corrections on the offset info, so do so here:
    sequence_file = os.path.join(fdir, fname + '_mc.sima/sequences.pkl')
    sequence_data = pickle.load(open(sequence_file, "rb"))  # load the saved sequences pickle file
    if bidi_offset > 0:
        sequence_data[0]['base']['displacements'][:, 0, 1::2,
        1] += bidi_offset  # add bidi shift to existing offset values
    else:  # can't have negative shifts in sima sequence, so have to offset even rows the opposite direction
        sequence_data[0]['base']['displacements'][:, 0, ::2, 1] -= bidi_offset
    with open(sequence_file, 'wb') as handle:
        pickle.dump(sequence_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def fill_gaps(framenumber, sequence, frame_iter1):  # adapted from SIMA source code/Vijay Namboodiri
    first_obs = next(frame_iter1)
    for frame in frame_iter1:
        for frame_chan, fobs_chan in zip(frame, first_obs):
            fobs_chan[np.isnan(fobs_chan)] = frame_chan[np.isnan(fobs_chan)]
        if all(np.all(np.isfinite(chan)) for chan in first_obs):
            break
    most_recent = [x * np.nan for x in first_obs]
    while True:
        frame = np.array(sequence[framenumber])[0, :, :, :, :]
        for fr_chan, mr_chan in zip(frame, most_recent):
            mr_chan[np.isfinite(fr_chan)] = fr_chan[np.isfinite(fr_chan)]
        temp = [np.nan_to_num(mr_ch) + np.isnan(mr_ch) * fo_ch
                for mr_ch, fo_ch in zip(most_recent, first_obs)]
        framenumber = yield np.array(temp)[0, :, :, 0]


def full_process(fpath, fparams):

    """

    important note: sima saves a folder (.sima) that contains a sequences pickle file. This file contains the offsets
    calculated from the motion correction algorithm. Sima by itself does not save a new video/tiff dataset that is motion
    corrected.

    :param fpath:
    :param max_disp:
    :param save_displacement:
    :return:
    """

    if 'flag_bidi_corr' not in fparams:
        fparams['flag_bidi_corr'] = True
    if 'flag_save_displacement' not in fparams:
        fparams['save_displacement'] = False
    if 'flag_save_h5' not in fparams:
        fparams['flag_save_h5'] = False
    if 'flag_save_projections' not in fparams:
        fparams['flag_save_projections'] = False

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
        start_time = time.time()
        # define motion correction method
        # n_processes can only handle =1! Bug in their code where >1 runs into an error
        # max_displacement: The maximum allowed displacement magnitudes in pixels in [y,x]
        mc_approach = sima.motion.HiddenMarkov2D(granularity='row', max_displacement=fparams['max_disp'], n_processes=1, verbose=True)

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
        filled_data = None # clear filled_data intermediate variable
        data_mc = np.squeeze(data_mc)

        if fparams['flag_save_displacement'] is True:
            # show motion displacements after motion correction
            mcDisp_approach = sima.motion.HiddenMarkov2D(granularity='row', max_displacement=fparams['max_disp'], n_processes=1,
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

        end_time = time.time()
        print("Motion correction execution time: {} seconds".format(end_time - start_time))

        if fparams['flag_bidi_corr']:
            start_time = time.time()
            # perform bidirection offset correction
            my_bidi_corr_obj = bidi_offset_correction.bidi_offset_correction(data_mc)  # initialize data to object

            my_bidi_corr_obj.compute_mean_image()  # compute mean image across time
            bidi_offset = my_bidi_corr_obj.determine_bidi_offset()  # calculated bidirectional offset via fft cross-correlation

            apply_bidi_corr_to_sima_offsets(fdir, fname, bidi_offset)

            end_time = time.time()
            print("Bidi offset correction execution time: {} seconds".format(end_time - start_time))


        # save motion-corrected, bidi offset corrected dataset
        if fparams['flag_save_h5']:

            dataset = sima.ImagingDataset.load(os.path.join(fdir, os.path.splitext(fname)[0] + '_mc.sima'))
            sequence_data = dataset.sequences[0]

            data_to_save = np.empty([dataset.num_frames, dataset.frame_shape[1], dataset.frame_shape[2]])
            frame_iter1 = iter(sequence_data)

            fill_gapscaller = fill_gaps(0, sequence_data, frame_iter1)
            fill_gapscaller.send(None)

            for frame_num in range(dataset.num_frames):
                data_to_save[frame_num, ...] = fill_gapscaller.send(frame_num).astype('int16')

            sima_mc_bidi_outpath = os.path.join(fdir, fname + '_sima_mc.h5')
            h5_write_bidi_corr = h5py.File(sima_mc_bidi_outpath, 'w')
            h5_write_bidi_corr.create_dataset('imaging', data=data_to_save)
            h5_write_bidi_corr.close()

        # save raw and MC mean images as figure
        # save_mean_imgs(save_dir, np.array(sequences), data_out)
        # calculate and save projection images and save as tiffs
        if fparams['flag_save_projections']:
            save_projections_chunked(fdir, fname, save_dir)



