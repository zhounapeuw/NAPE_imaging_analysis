#!/usr/bin/env python
# coding: utf-8

# # Bruker raw ome-tiff preparation for preprocessing pipeline

import numpy as np
import os
import glob
from scipy import signal
import h5py
import warnings
import multiprocessing as mp

import matplotlib.pyplot as plt
from PIL import Image
from PIL.TiffTags import TAGS
import tifffile as tiff
from lxml.html.soupparser import fromstring
from lxml.etree import tostring

from copy import copy, deepcopy


# function to load tiff data and get data shape
def uint16_scale(img):
    tmp = img - np.min(img) # shift values such that there are no negatives

    ratio = np.amax(tmp) / 65535.0

    return np.squeeze(tmp/ratio) 


def read_shape_tiff(data_path):
    
    data = uint16_scale(tiff.imread(data_path)).astype('uint16')
    data_shape = data.shape

    return data, data_shape


def get_tif_meta(tif_path):
    meta_dict = {}
    # iterate through metadata and create dict for key/value pairs
    with Image.open(tif_path) as img:
        for key in img.tag.iterkeys():
            if key in TAGS:
                meta_dict[TAGS[key]] = img.tag[key] 
            else:
                meta_dict[key] = img.tag[key] 
    
    return meta_dict


def check_if_meta_tif(path):
    
    meta_dict = get_tif_meta(path)

    # 'ImageDescription' key contains info about the file(s) in xml format   
    tag_soup = str(meta_dict['ImageDescription'][0][21:])
    root_meta = fromstring(tag_soup) # process xml string

    # the 'image' tag in the xml is unique to the first tif with metadata; check for that
    subdict_image = []
    for neighbor in root_meta.iter('image'):
         subdict_image = neighbor.attrib

    return 'id' in subdict_image


def threshold_img(data_thresh, thresh_percent=2):
    
    # get ranges for values across whole dataset and set a threshold to turn low amplitude noise to 0
    data_range = [np.min(data_thresh), np.max(data_thresh)]
    threshold_set_0 = data_range[1]*(thresh_percent/100.0) # set all values less than 1% of max signal to 0 with this threshold


def assert_bruker(fpath):
    meta_dict = get_tif_meta(fpath)
    assert ('Prairie' in meta_dict['Software'][0]), "This is not a bruker file!"


########## USER DEFINE VARIABLES

fname = 'vj_ofc_imageactivate_001_20200813-014'
root_session_folder = r'D:\bruker_data\Charles\vj_ofc_imageactivate_001_20200813\vj_ofc_imageactivate_001_20200813-014'

flag_save_type = 'h5' # set is 'tif' or 'h5'. SELECT h5 if file size will be larger than 4 gb!!!! Sima can't handle bigtiffs
#num_frames = 3600 # optional; number of frames to analyze; defaults to analyzing whole session


if __name__ == "__main__":

    save_fname = os.path.join(root_session_folder, fname)
    glob_list = glob.glob(os.path.join(root_session_folder,"*.tif"))
    #if not 'num_frames' in locals(): # CZ tmp: comment back in once make this into a function
    num_frames = len(glob_list)
    displayed_slice = np.random.choice(num_frames)
    print(str(num_frames) + ' total frame')

    # prepare to split data into chunks when loading to reduce memory imprint
    chunk_size = 10000.0
    n_chunks = int(np.ceil(num_frames/chunk_size))
    print(str(n_chunks) + ' chunks')
    chunked_frame_idx = np.array_split(np.arange(num_frames), n_chunks) # split frame indices into chunks

    # read first tiff to get data shape
    first_tif = tiff.imread(glob_list[0], key=0, is_ome=True)
    frame_shape = first_tif.shape

    assert_bruker(glob_list[0])
    print('Processing Bruker data')

    # prepare handles to write data to
    if flag_save_type == 'tif':
        f = tiff.TiffWriter(save_fname + '.tif', bigtiff=True)
    elif flag_save_type == 'h5':
        f = h5py.File(save_fname + '.h5', 'w')
        # get data shape and chunk up data, and initialize h5
        dset = f.create_dataset('imaging', (num_frames, frame_shape[0], frame_shape[1]),
                                maxshape=(None, frame_shape[0], frame_shape[1]), dtype='uint16')

    # go through each chunk, load frames in chunk, process, and append to file
    for idx, chunk_frames in enumerate(chunked_frame_idx):
        print('Processing chunk {}'.format(str(idx)))
        start_idx = chunk_frames[0]
        end_idx = chunk_frames[-1]+1

        #loaded_tiffs = uint16_scale(tiff.imread(glob_list[start_idx:end_idx], key=0, is_ome=True))
        loaded_tiffs = tiff.imread(glob_list[start_idx:end_idx], key=0, is_ome=True)

        if flag_threshold_LUT:
            data_to_save = threshold_img(loaded_tiffs, thresh_percent=LUT_lower_thresh) # turn data to uint16 first, then threshold LUT
        else:
            data_to_save = loaded_tiffs

        if flag_save_type == 'tif':

            for frame in tiffs_to_save:
                f.save(frame, photometric='minisblack')

        # https://stackoverflow.com/questions/25655588/incremental-writes-to-hdf5-with-h5py
        elif flag_save_type == 'h5':

            # append data to h5
            dset[start_idx:end_idx] = data_to_save

    if flag_save_type == 'h5':
        f.close()





