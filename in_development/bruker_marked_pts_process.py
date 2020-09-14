#!/usr/bin/env python
# coding: utf-8


"""
The bruker scope turns off the PMT during stimulation times, so fluorescence on certain lines are balnked. Using a combination of setting a threshold for the pixel-averaged fluorescence time-series and stim times from analog ttl (extracted using bruker_data_process), identify the frames that contain stim.

Also plots the mark points stim ROIs on the mean image

Currently looks for "stim" keys in the analog event dictionary for analog signals that represent stimulation

"""

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xml.etree.ElementTree as ET
import pickle
import pandas as pd

import utils_bruker


### Loading functions

def load_ca_data(fdir, fname):
    h5_file = h5py.File(os.path.join(fdir, fname + '.h5'), 'r')
    return h5_file.get(list(h5_file)[0])[()] # [()] grabs the values


# takes bruker marked points xml data, goes through each iteration, group, and point and grabs meta data
# all times are in ms
def load_mark_pt_xml_df(path_vars, im_shape):
    
    mark_pt_xml_parse = ET.parse(path_vars['mark_pt_xml_path']).getroot()
    mk_pt_dict = {'iterations': int(mark_pt_xml_parse.attrib['Iterations']), 
                  'iter_delay': float(mark_pt_xml_parse.attrib['IterationDelay'])}

    mk_pt_df = pd.DataFrame()
    point_counter = 0
    for type_tag in mark_pt_xml_parse.findall('PVMarkPointElement'):

        laser_pow = float(type_tag.attrib['UncagingLaserPower'])*100
        reps = int(type_tag.attrib['Repetitions'])

        for group_tag in type_tag.findall('PVGalvoPointElement'):

            duration = float(group_tag.attrib['Duration'])
            IPI = float(group_tag.attrib['InterPointDelay'])
            initial_delay = float(group_tag.attrib['InitialDelay'])
            try:
                group = group_tag.attrib['Points']
            except:
                print('No Group')

            for point in group_tag.findall('Point'):

                mk_pt_df.loc[point_counter, 'group'] = group
                mk_pt_df.loc[point_counter, 'repetitions'] = reps
                mk_pt_df.loc[point_counter, 'height'] = np.round(float(point.attrib['SpiralHeight'])*im_shape[0])
                mk_pt_df.loc[point_counter, 'width'] = np.round(float(point.attrib['SpiralWidth'])*im_shape[1])
                mk_pt_df.loc[point_counter, 'IsSpiral'] = point.attrib['IsSpiral']
                mk_pt_df.loc[point_counter, 'Y'] = np.round(float(point.attrib['Y'])*im_shape[0])
                mk_pt_df.loc[point_counter, 'X'] = np.round(float(point.attrib['X'])*im_shape[1])
                mk_pt_df.loc[point_counter, 'duration'] = duration
                mk_pt_df.loc[point_counter, 'IPI'] = IPI
                mk_pt_df.loc[point_counter, 'initial_delay'] = initial_delay
                mk_pt_df.loc[point_counter, 'pow'] = laser_pow
                point_counter += 1
    
    return mk_pt_df


# loads dict of analog events dict
def load_analog_stim_samples(analog_event_path):
    if os.path.exists(analog_event_path):
        with open(analog_event_path, 'rb') as handle:
            analog_event_dict = pickle.load(handle)

        if 'stim' in analog_event_dict.keys():
            return np.array(list(set(analog_event_dict['stim'])))
        else:
            return []
    else:
        return []


### analysis functions

# take avg fluorescene across pixels and take threshold
def std_thresh_stim_detect(im, thresh_std=2.5): 
    im_pix_avg = np.squeeze(np.mean(im, axis=(1,2)))
    im_pix_avg_std = np.std(im_pix_avg)
    im_pix_avg_avg = np.mean(im_pix_avg)
    thresh = im_pix_avg_avg - im_pix_avg_std*thresh_std

    return np.where(im_pix_avg < thresh)[0], thresh


### plotting functions

# plot pix-avg t-series of video, blanked frames t-series, and threshold 
def plot_blanked_frames(im_pix_avg, stimmed_frames, thresh_val, lims = None):
    
    im_pix_avg_copy = np.copy(im_pix_avg)
    im_pix_avg_copy[stimmed_frames['samples']] = np.nan
    fig, ax = plt.subplots(1,1)
    ax.plot(im_pix_avg)
    ax.plot(im_pix_avg_copy)
    ax.plot(np.ones(len(im_pix_avg))*thresh_val)
    if lims:
        ax.set_xlim(lims)
    ax.legend(['original', 'blanked_stim', 'threshold'])
    
    
def plot_stim_locations(im, path_vars):
    img_mean = np.mean(im, axis=0)
    img_mean_clims = [np.min(img_mean)*1.2, np.max(img_mean)*0.6]

    mk_pt_df = load_mark_pt_xml_df(path_vars, img_mean.shape)

    point_counter = 0
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.imshow(img_mean, clim=img_mean_clims, cmap='gray')
    plot_mk_pts = True
    if plot_mk_pts:
        for idx, row in mk_pt_df.iterrows():
            mk_pt_ellipse = matplotlib.patches.Ellipse((row['X'], row['Y']),
                                       row['height'], row['width'])
            ax.add_artist(mk_pt_ellipse)
            mk_pt_ellipse.set_clip_box(ax.bbox)
            mk_pt_ellipse.set_edgecolor(np.random.rand(3))
            mk_pt_ellipse.set_facecolor('None')

        ax.axis('off')


# In[14]:


def main_detect_save_stim_frames(fdir, fname, detection_threshold=1.5, flag_plot_mk_pts=False):

    path_vars = {}
    path_vars['tseries_xml_path'] = os.path.join(fdir, fname + '.xml')
    path_vars['mark_pt_xml_path'] = os.path.join(fdir, fname + '_Cycle00001_MarkPoints.xml')
    path_vars['analog_event_path'] = os.path.join(fdir, 'framenumberforevents_{}.pkl'.format(fname))

    path_vars['mk_pt_h5_savepath'] = os.path.join(fdir, fname + 'mk_pt_meta.h5')
    path_vars['stim_frames_savepath'] = os.path.join(fdir, fname + '_stimmed_frames.pkl')

    fs_2p = utils_bruker.bruker_xml_get_2p_fs(path_vars['tseries_xml_path'])
    im = load_ca_data(fdir, fname)

    # load, analyze, and combine detected stim frames
    analog_detected_stims = load_analog_stim_samples(path_vars['analog_event_path'])
    thresh_detected_stims, thresh_val = std_thresh_stim_detect(im, thresh_std=detection_threshold)
    analog_thresh_detected_stims = np.union1d(analog_detected_stims, thresh_detected_stims).astype('int')

    # add stimmed frames to dict
    stimmed_frames = {}
    stimmed_frames['samples'] = analog_thresh_detected_stims
    stimmed_frames['times'] = analog_thresh_detected_stims/fs_2p

    # save pickled dict that contains frames and corresponding frame times where pulses occurred
    with open(path_vars['stim_frames_savepath'], 'wb') as handle:
        pickle.dump(stimmed_frames, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plot pix-avg t-series, t-series with blanked frames, and threshold
    im_pix_avg = np.squeeze(np.nanmean(im, axis=(1,2)))
    plot_blanked_frames(im_pix_avg, stimmed_frames, thresh_val, lims = None)

    # plot mark point stim locations on mean img
    if flag_plot_mk_pts:
        plot_stim_locations(im, path_vars)


if __name__ == "__main__":

    fname = 'vj_ofc_imageactivate_001_20200828-003'
    fdir = r'D:\bruker_data\vj_ofc_imageactivate_001_20200828\vj_ofc_imageactivate_001_20200828-003'

    detection_threshold = 1.5
    flag_plot_mk_pts = True

    main_detect_save_stim_frames(fdir, fname, detection_threshold, flag_plot_mk_pts)


# In[ ]:




