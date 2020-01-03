#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import scipy.io as sio
import os
import subprocess
import bisect
import errno
import time
import pandas
import pickle
import num2word
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score
#from sklearn import cross_validation
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import scipy.stats as stats
from sklearn.metrics import roc_auc_score as auROC
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import (ModelDesc, EvalEnvironment, Term, EvalFactor, LookupFactor, dmatrices, INTERCEPT)
from statsmodels.distributions.empirical_distribution import ECDF
from shapely.geometry import MultiPolygon, Polygon, Point
import PIL
from itertools import product
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
from sima.ROI import poly2mask, _reformat_polygons
import h5py
import sima
import sys
from scipy import optimize
from multiprocessing import Pool, cpu_count

import rasterio.features
import shapely.geometry
import matplotlib.pyplot as plt


# In[6]:


filename = 'itp_lhganiii_p3ml_830p8_tiff'
root_dir = 'C:\\2pData\\Ivan\\itp_lhganiii_p3ml_830p8\\'
suite2p_dat_dir = root_dir + 'suite2p\\plane0\\'
suite2p_h5_path = root_dir + filename + '_suite2p_mc.h5'
sima_h5_path = root_dir + filename + '_sima_mc.h5'

fig_save_dir = root_dir + 'figs\\'
proj_dir = root_dir + "projections\\"

# projections path
fpath = proj_dir + filename + '_projections'
hf = h5py.File(fpath, 'r')

list(hf)


# In[49]:


# these two functions are for loading ROI masks from preprocessing packages and obtaining ROI attributes

# embedded within: calculate_roi_centroids
def load_rois_for_session(session_folder, package):


    # load s2p meta data and load cells
    suite2p_dat_dir = session_folder + '\\suite2p\\plane0\\'
    stat = np.load( suite2p_dat_dir + 'stat.npy', allow_pickle=True)
    ops = np.load( suite2p_dat_dir + 'ops.npy', allow_pickle=True).item()
    iscell = np.load(suite2p_dat_dir + "iscell.npy", allow_pickle = True)
    cell_ids = np.where( iscell[:,0] == 1 )[0]  # indices of detected cells across all ROIs from suite2p
    num_cells_suite2p = len(cell_ids)

    roi_polygons = []
    for iROI in cell_ids:

        # make binary ROI mask
        roi_mask = np.zeros([ops['Ly'],ops['Lx']], dtype = 'float32')
        roi_mask[stat[iROI]['ypix'], stat[iROI]['xpix']] = 1

        shapes = rasterio.features.shapes(roi_mask)
        polygons = [shapely.geometry.Polygon(shape[0]["coordinates"][0]) for shape in shapes if shape[1] == 1]

        roi_polygons.append(polygons[0])
        im_shape = (ops['Ly'],ops['Lx'])
            
    return roi_polygons, im_shape #CZ roi_polygona

# 3RD FUNCTION IN ORDER
def calculate_roi_centroids(session_folder):
   
    stat = np.load( suite2p_dat_dir + 'stat.npy', allow_pickle=True)
    ops = np.load( suite2p_dat_dir + 'ops.npy', allow_pickle=True).item()
    stat[0]['med'] # ROI center; basically median for each dimension
    
    return roi_centroids, im_shape, roi_polygons


# In[50]:


# 4TH FUNCTION IN ORDER
def calculate_roi_masks(roi_polygons, im_size):
    masks = []
    if len(im_size) == 2:
        im_size = (1,) + im_size
    roi_polygons = _reformat_polygons(roi_polygons)
    for poly in roi_polygons:
        mask = np.zeros(im_size, dtype=bool)
        # assuming all points in the polygon share a z-coordinate
        z = int(np.array(poly.exterior.coords)[0][2])
        if z > im_size[0]:
            warn('Polygon with zero-coordinate {} '.format(z) +
                 'cropped using im_size = {}'.format(im_size))
            continue
        x_min, y_min, x_max, y_max = poly.bounds

        # Shift all points by 0.5 to move coordinates to corner of pixel
        shifted_poly = Polygon(np.array(poly.exterior.coords)[:, :2] - 0.5)

        points = [Point(x, y) for x, y in
                  product(np.arange(int(x_min), np.ceil(x_max)),
                          np.arange(int(y_min), np.ceil(y_max)))]
        points_in_poly = list(filter(shifted_poly.contains, points))
        for point in points_in_poly:
            xx, yy = point.xy
            x = int(xx[0])
            y = int(yy[0])
            if 0 <= y < im_size[1] and 0 <= x < im_size[2]:
                mask[z, y, x] = True
        masks.append(mask[0,:,:])
    return masks

# 5TH FUNCTION
def calculate_spatialweights_around_roi(indir, roi_masks, roi_centroids,
                                        neuropil_radius, min_neuropil_radius, h5file):
    
    # roi_centroids has order (x,y). The index for any roi_masks is in row, col shape or y,x shape.
    # So be careful to flip the order when you subtract from centroid
    numrois = len(roi_masks)
    allrois_mask = np.logical_not(np.sum(roi_masks, axis=0))
    (im_ysize, im_xsize) = allrois_mask.shape
    y_base = np.tile(np.array([range(1,im_ysize+1)]).transpose(), (1,im_xsize))
    x_base = np.tile(np.array(range(1,im_xsize+1)), (im_ysize,1))
    
    # Set weights for a minimum radius around all ROIs to zero as not the whole ROI is drawn
    deadzones_aroundrois = np.ones((im_ysize, im_xsize))
    for roi in range(numrois):
        x_diff = x_base-roi_centroids[roi][0]
        y_diff = y_base-roi_centroids[roi][1]
        dist_from_centroid = np.sqrt(x_diff**2 + y_diff**2)
        temp = np.ones((im_ysize, im_xsize))
        temp[dist_from_centroid<min_neuropil_radius] = 0
        deadzones_aroundrois *= temp
    
    allrois_mask *= deadzones_aroundrois.astype(bool)
    
    # initialize H5 to save spatial weight info
    h5 = h5py.File(os.path.join(indir, '%s_spatialweights_%d_%d.h5'%(os.path.splitext(h5file)[0],
                                                                     min_neuropil_radius,
                                                                     neuropil_radius)),
                   'w', libver='latest')
    
    output_shape = (numrois, im_ysize, im_xsize)
    h5['/'].create_dataset(
        'spatialweights', output_shape, maxshape=output_shape,
        chunks=(1, output_shape[1], output_shape[2]))
    
    h5['/'].create_dataset('deadzones_aroundrois',data = deadzones_aroundrois)
    
    # go through each roi and calculate spatial weight map for neuropils
    for roi in range(numrois):
        x_diff = x_base-roi_centroids[roi][0]
        y_diff = y_base-roi_centroids[roi][1]
        dist_from_centroid = np.sqrt(x_diff**2 + y_diff**2)
        spatialweights = np.exp(-(x_diff**2 + y_diff**2)/neuropil_radius**2)
        spatialweights *= im_ysize*im_xsize/np.sum(spatialweights)
        
        # Set weights for a minimum radius around the ROI to zero
        #spatialweights[dist_from_centroid<min_neuropil_radius] = 0
        # Set weights for pixels containing other ROIs to 0
        spatialweights *= allrois_mask
        """fig, ax = plt.subplots()
        ax.imshow(spatialweights, cmap='gray')
        raise Exception()"""
        h5['/spatialweights'][roi, :, :] = spatialweights
     
    h5.close()
    
def calculate_neuropil_coefficients_for_session(indir, signals, neuropil_signals,
                                                neuropil_radius, min_neuropil_radius, beta_neuropil=None):
    
    skewness_rois = np.nan*np.ones((signals.shape[0],2)) #before, after correction
    if beta_neuropil is None:
        beta_rois = np.nan*np.ones((signals.shape[0],))
        for roi in range(signals.shape[0]):        
            def f(beta):
                temp1 = signals[roi]-beta*neuropil_signals[roi]
                temp2 = neuropil_signals[roi]
                _,_,_,_,temp3 = fit_regression(temp1, temp2)
                return temp3
            
            beta_rois[roi] = optimize.minimize(f, [1], bounds=((0,None),)).x
            skewness_rois[roi,0] = stats.skew(signals[roi])
            temp1 = signals[roi]-beta_rois[roi]*neuropil_signals[roi]
            temp2 = neuropil_signals[roi]
            _,temp4,_,_,temp3 = fit_regression(temp1, temp2)
            skewness_rois[roi,1] = np.sqrt(temp3)* np.sign(temp4)

        #print beta_rois
        fig, axs = plt.subplots(1,2,figsize=(8,4))
        CDFplot(beta_rois, axs[0])
        CDFplot(skewness_rois[:,1], axs[1])
        
        return beta_rois, skewness_rois
    else:
        skewness_rois[:,0] = stats.skew(signals, axis=1)
        skewness_rois[:,1] = stats.skew(signals-beta_neuropil*neuropil_signals, axis=1)

        return beta_neuropil, skewness_rois

def save_neuropil_corrected_signals(indir, signals, neuropil_signals, beta_rois,
                                    neuropil_radius, min_neuropil_radius, h5file):
    
    #CZ tmp
    print('signals {} beta ROIs {} npil_sigs {}'.format(str(signals.shape),str(beta_rois),str(neuropil_signals.shape)))
    
    # beta ROIs come from: calculate_neuropil_coefficients_for_session
    if isinstance(beta_rois, np.ndarray):
        corrected_signals = signals-beta_rois[:,None]*neuropil_signals
        np.save(os.path.join(indir, '%s_neuropil_corrected_signals_%d_%d_betacalculated.npy'%(os.path.splitext(h5file)[0],
                                                                                           min_neuropil_radius,
                                                                                           neuropil_radius)),
                corrected_signals)
    else:
        corrected_signals = signals-beta_rois*neuropil_signals
        np.save(os.path.join(indir, '%s_neuropil_corrected_signals_%d_%d_beta_%.1f.npy'%(os.path.splitext(h5file)[0],
                                                                                         min_neuropil_radius,
                                                                                         neuropil_radius,
                                                                                         beta_rois)),
                corrected_signals) 


# In[14]:


# SECOND EMBEDDED NEUROPIL FUNCTION 
def calculate_neuropil_signals(h5filepath, neuropil_radius, min_neuropil_radius,
                               masked=False):
    
    dual_channel = False
    
    savedir = os.path.dirname(h5filepath)
    h5file = h5py.File(h5filepath,'r') #Read-only
    h5filename = os.path.basename(h5filepath)
    
    simadir = os.path.splitext(h5filename)[0]+'_mc.sima'
    
    dataset = sima.ImagingDataset.load(os.path.join(savedir, simadir))
    sequence = dataset.sequences[0]
    frame_iter1 = iter(sequence)
    
    def fill_gaps(framenumber):  #adapted from SIMA source code  
        first_obs = next(frame_iter1)
        for frame in frame_iter1:
            for frame_chan, fobs_chan in zip(frame, first_obs):
                fobs_chan[np.isnan(fobs_chan)] = frame_chan[np.isnan(fobs_chan)]
            if all(np.all(np.isfinite(chan)) for chan in first_obs):
                break
        most_recent = [x * np.nan for x in first_obs]
        while True:
            frame = np.array(sequence[framenumber])[0,:,:,:,:]
            for fr_chan, mr_chan in zip(frame, most_recent):
                mr_chan[np.isfinite(fr_chan)] = fr_chan[np.isfinite(fr_chan)]
            temp=[np.nan_to_num(mr_ch) + np.isnan(mr_ch) * fo_ch
                for mr_ch, fo_ch in zip(most_recent, first_obs)]
            framenumber = yield np.array(temp)[0,:,:,0]


    fill_gapscaller = fill_gaps(0)
    fill_gapscaller.send(None)
    
    roi_centroids, im_shape, roi_polygons = calculate_roi_centroids(savedir)
    roi_masks = calculate_roi_masks(roi_polygons, im_shape)
    
    calculate_spatialweights_around_roi(savedir, roi_masks, roi_centroids,
                                            neuropil_radius, min_neuropil_radius, h5filename)
    
    # load the spatial weights calculated from previous line
    h5weights = h5py.File(os.path.join(savedir, '%s_spatialweights_%d_%d.h5'%(os.path.splitext(h5filename)[0],
                                                                           min_neuropil_radius, neuropil_radius)), 'r')
    
    spatialweights = h5weights['/spatialweights']
    
    numframes = h5file['/imaging'].shape[0]
    neuropil_signals = np.nan*np.ones((len(roi_masks), numframes))
    
    #pb = ProgressBar(numframes)
    start_time = time.time()
    for frame in range(numframes):
        
        temp = fill_gapscaller.send(frame)[None,:,:] #this will fill gaps in rows by interpolation
        neuropil_signals[:, frame] = np.einsum('ijk,ijk,ijk->i', spatialweights,
                                               temp, np.isfinite(temp))#/np.sum(spatialweights, axis=(1,2))
        # The einsum method above is way faster than multiplying array elements individually
        # The above RHS basically implements a nanmean and averages over x and y pixels
        
        #pb.animate(frame+1)
    neuropil_signals /= np.sum(spatialweights, axis=(1,2))[:,None]  
    
    np.save(os.path.join(savedir, '%s_neuropilsignals_%d_%d.npy'%(os.path.splitext(h5filename)[0],
                                                           min_neuropil_radius,
                                                           neuropil_radius)),
        neuropil_signals)
    
# MAIN FUNCTION 
def calculate_neuropil_signals_for_session(indir, h5filename, neuropil_radius=50,
                                           min_neuropil_radius=15, beta_neuropil=0.8,
                                           masked=False):
    
    # masked refers to whether any frames have been masked due to light artifacts
    print('indir: ' + indir)
    
    
    # function for calculating neuropil time-series for each ROI
    # makes a deadzone around ROI, then gaussian-weighted area around deadzone 
    # this weighted area excludes the ROI masks of other cells
    calculate_neuropil_signals(os.path.join(indir, h5filename), neuropil_radius,
                               min_neuropil_radius, masked=masked)
    
    savedir = indir
    simadir = os.path.splitext(h5filename)[0]+'_mc.sima'
    npyfile = os.path.splitext(h5filename)[0]+'_sima_extractedsignals.npy'
    signals = np.squeeze(np.load(os.path.join(indir, npyfile)))
    
    # calculate central coordinates of ROIs
    roi_centroids, im_shape, roi_polygons = calculate_roi_centroids(savedir)

    dataset = sima.ImagingDataset.load(os.path.join(savedir, simadir))
    
    # calculate ROI mask and corresponding pixel-avg time-series
    roi_masks = calculate_roi_masks(roi_polygons, im_shape)
    
    # dataset.time_averages is time-avg image?
    # multiply roi binary mask with mean image, sum across pixels, then divide by num pixels (mean)
    # mean_roi_response is a vector of mean ROI fluorescence values
    mean_roi_response = np.nansum(roi_masks*dataset.time_averages[:,:,:,0], axis=(1,2))/np.sum(roi_masks, axis=(1,2))
    
    # sima divides ROI time-series by the mean response; reverse this
    signals *= mean_roi_response[:,None]
    
    np.save(os.path.join(savedir, '{}_mean_roi_resp'.format(os.path.splitext(h5filename)[0])),
        mean_roi_response)
    
    # load neuropil signals
    neuropil_signals = np.squeeze(np.load(os.path.join(indir,
                                                       '%s_neuropilsignals_%d_%d.npy'%(os.path.splitext(h5filename)[0],
                                                                                       min_neuropil_radius, 
                                                                                       neuropil_radius))))
                                                                     
    beta_rois, skewness_rois = calculate_neuropil_coefficients_for_session(indir, signals, neuropil_signals, 
                                                                           neuropil_radius, min_neuropil_radius,
                                                                           beta_neuropil=beta_neuropil)
                                                                                       
    save_neuropil_corrected_signals(indir, signals, neuropil_signals, beta_rois,
                                    neuropil_radius, min_neuropil_radius, h5filename)


# In[ ]:


if 'h5weights' in locals():
    h5weights.close() # sometime error arises if prior h5 file opening isn't properly closed

calculate_neuropil_signals_for_session(root_dir, filename + '.h5', neuropil_radius=50,
                                           min_neuropil_radius=15, beta_neuropil=0.8,
                                           masked=False)


# # Plot Data

# In[69]:


# load spatial weights
h5filename = filename + '.h5'
neuropil_radius=50
min_neuropil_radius=15
h5weights = h5py.File(os.path.join(root_dir, '%s_spatialweights_%d_%d.h5'%(os.path.splitext(h5filename)[0],
                                                                           min_neuropil_radius, neuropil_radius)), 'r')

masks = np.load(os.path.join(root_dir,'itp_lhganiii_p3ml_830p8_tiff_sima_masks.npy'))
extract_signals = np.load(os.path.join(root_dir,'itp_lhganiii_p3ml_830p8_tiff_sima_extractedsignals.npy'))
mean_roi_resp = np.load(os.path.join(root_dir,'itp_lhganiii_p3ml_830p8_tiff_mean_roi_resp.npy'))
npil_signals = np.load(os.path.join(root_dir,'itp_lhganiii_p3ml_830p8_tiff_neuropilsignals_15_50.npy'))
signals_npil_corr = np.load(os.path.join(root_dir,'itp_lhganiii_p3ml_830p8_tiff_neuropil_corrected_signals_15_50_beta_0.8.npy'))


# In[88]:


iROI = 10


# In[89]:


# plot each ROI's cell mask
to_plot = masks[iROI,:,:] # single ROI
to_plot = np.sum(masks, axis = 0) # all ROIs

plt.figure()
plt.imshow(to_plot, cmap = 'gray')
plt.title('Single ROI Mask', fontsize = 20)
plt.axis('off');


# In[90]:


to_plot = h5weights['deadzones_aroundrois']

plt.figure()
plt.imshow(to_plot, cmap = 'gray')
plt.title('ROI Soma Deadzones', fontsize = 20);
plt.tick_params(labelleft=False, labelbottom=False) 


# In[91]:


# plot the gaussian spatial weights around an ROI

plt.figure()
to_plot = h5weights['spatialweights'][iROI,:,:]
plt.imshow(to_plot, cmap = 'gray')
plt.title('Neuropil Spatial Weights', fontsize = 20)
plt.axis('off');


# In[92]:


# function to z-score time series
z_score = lambda sig_in: (sig_in - np.mean(sig_in))/np.std(sig_in)


# In[93]:


# plot the ROI pixel-avg signal, npil signal, and npil corrected ROI signal

fig, ax = plt.subplots(1,2, figsize = (15,5))
ax[0].plot( z_score(extract_signals[0,iROI,:]), alpha = 0.8 )
ax[0].plot( z_score(signals_npil_corr[iROI,:]), alpha = 0.5 )
ax[0].legend(['Extracted sig', 'Npil-corr Sig'], fontsize = 15);
ax[0].set_xlabel('Time [s]', fontsize = 15);
ax[0].set_ylabel('Normalized Fluorescence', fontsize = 15);

ax[1].plot( npil_signals[iROI,:], alpha = 0.6 )
ax[1].legend(['Neuropil Sig'], fontsize = 15);
#plt.xlim([0,2000])


# In[94]:


beta_npil = 0.8

plt.figure()
plt.plot( extract_signals[0,iROI,:]*mean_roi_resp[iROI], alpha = 0.6 )
plt.plot( npil_signals[iROI,:]*beta_npil, alpha = 0.6)
plt.plot( signals_npil_corr[iROI,:], alpha = 0.6)
plt.title('ROI ' + str(iROI), fontsize = 20)
plt.legend(['sima extracted sig','npil sig', 'npil-corr sig']);
#plt.xlim([0,2000])


# In[ ]:




