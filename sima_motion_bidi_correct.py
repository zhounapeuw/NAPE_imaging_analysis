import sima
import sima.motion
import numpy as np
import os
import pickle
import h5py
import sys
sys.path.insert(1, 'C:\\Users\\stuberadmin\\Dropbox (Stuber Lab)\\Python\\Charles\\')
import bidi_offset_correction
from contextlib import contextmanager

def unpack(args):
    print(args)
    return sima_motion_correction(*args)

def sima_motion_correction(fpath, max_disp, save_displacement=False):
    print('sima_motion_correction')
    fdir  = os.path.split(fpath)[0]
    fname = os.path.splitext(os.path.split(fpath)[1])[0]
    fext  = os.path.splitext(os.path.split(fpath)[1])[1]

    if fext == '.tif' or fext == '.tiff':
        # sequence: object that contains record of whole dataset; data not stored into memory all at once
        sequences = [sima.Sequence.create('TIFF', fpath)]

    elif fext == '.h5':
        sequences = [sima.Sequence.create('HDF5', fpath, 'tyx')]

    # define motion correction method
    # n_processes can only handle =1! Bug in their code where >1 runs into an error
    # max_displacement: The maximum allowed displacement magnitudes in pixels in [y,x]
    mc_approach = sima.motion.HiddenMarkov2D(granularity='row', max_displacement=max_disp, n_processes=1, verbose=True)

    # apply motion correction to data
    dataset, __, _ = mc_approach.correct(sequences, os.path.join(fdir, fname + '_mc.sima'), channel_names=['GCaMP'])
    # dataset dimensions are frame, plane, row(y), column (x), channel

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

    # convert motion corrected sequence to nparray
    data_mc = np.squeeze(np.array(dataset[0]._sequences[0]).astype('int16'))  # np.array loads all data into memory

    # perform bidirection offset correction
    my_bidi_corr_obj = bidi_offset_correction.bidi_offset_correction(data_mc)  # initialize data to object
    my_bidi_corr_obj.compute_mean_image()  # compute mean image across time
    my_bidi_corr_obj.determine_bidi_offset()  # calculated bidirectional offset via fft cross-correlation
    data_corrected = my_bidi_corr_obj.correct_bidi_frames()  # apply bidi offset to data

    # save motion-corrected, bidi offset corrected dataset
    sima_mc_bidi_outpath = os.path.join(fdir, fname + '_sima_mc.h5')
    h5_write_bidi_corr = h5py.File(sima_mc_bidi_outpath, 'w')
    h5_write_bidi_corr.create_dataset('imaging', data=data_corrected)
    h5_write_bidi_corr.close()
