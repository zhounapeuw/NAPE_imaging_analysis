#!/usr/bin/env python
# coding: utf-8

# In[52]:


import os
import numpy as np
import h5py
import pandas as pd
import pickle

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
plt.rcParams['text.latex.unicode'] = False


# In[53]:


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


# In[54]:


# load video data

# open h5 to read, find data key, grab data, then close
h5 = h5py.File(suite2p_h5_path,'r')
suite2p_data = np.squeeze(np.array( h5[list(h5.keys())[0]] )).astype('int16') # np.array loads all data into memory
h5.close()
num_samples = int(suite2p_data.shape[0])


# In[55]:


# load suite2p roi infor
ops = np.load(suite2p_dat_dir + "ops.npy", allow_pickle = True).item()
iscell = np.load(suite2p_dat_dir + "iscell.npy", allow_pickle = True)

# load suite2p roi time-series and stat has contours
F = np.load(suite2p_dat_dir + 'F.npy')
Fneu = np.load(suite2p_dat_dir + 'Fneu.npy')
stat = np.load(suite2p_dat_dir + 'stat.npy', allow_pickle = True)

print('stat dict keys:')
print(stat[0].keys())


# In[56]:


# initialize variables for plotting time-series
fs = ops['fs']
num_samps = ops['nframes']
total_time = num_samps/fs 
tvec = np.linspace(0,total_time,num_samps)


# In[57]:


# get all cell ROIs
cell_ids = np.where( iscell[:,0] == 1 )[0]  # indices of detected cells across all ROIs from suite2p
num_cells_suite2p = len(cell_ids)
print(num_cells_suite2p)

# define number of ROIs to visualize and make colormap
numROI_2Viz = num_cells_suite2p
cell_ids_analyze = cell_ids[:numROI_2Viz]
colors_roi = plt.cm.viridis(np.linspace(0,numROI_2Viz/6,numROI_2Viz))


# In[58]:


# initialize templates for contour map
plot_these_roi = []
roi_label_loc = []
zero_template_suite2p = np.zeros([ops['Ly'], ops['Lx']])

# loop through ROIs and add their spatial footprints to template
for idx,iROI in enumerate(cell_ids_analyze):
    
    # make binary map of ROI pixels
    zero_template_suite2p[ stat[iROI]['ypix'],stat[iROI]['xpix'] ] = 1*(idx+1)
    
    # just an array of cell ROI numbers for subsequent analyses
    plot_these_roi.append(iROI) # CZ do I need this?
    
    roi_label_loc.append( [np.min(stat[iROI]['ypix']), np.min(stat[iROI]['xpix'])] )


# In[59]:


# plot contours and cell numbers
fig, ax = plt.subplots(1, 1, figsize = (12,12))
ax.imshow(hf['std_img'], cmap = 'gray')
ax.axis('off')
ax.contour(zero_template_suite2p, colors = colors_roi);

for idx,iROI in enumerate(cell_ids_analyze): 
    ax.text(roi_label_loc[idx][1], roi_label_loc[idx][0],  str(idx), fontsize=13, color = 'white')
    
plt.title('Suite2p ROIs', fontsize = 20)
#plt.savefig(fig_save_dir + 'roi_contour_map.jpg')
#plt.savefig(fig_save_dir + 'roi_contour_map.eps', format='eps')


# # Extract Signals

# In[60]:


# for suite2p, go through each ROI and take mean across pixels

roi_signal_suite2p = np.empty([numROI_2Viz,num_samples])

for idx,iROI in enumerate(cell_ids_analyze):

    roi_signal_suite2p[idx,:] = np.mean(suite2p_data[:, stat[iROI]['ypix'], stat[iROI]['xpix']], 
                         axis = 1)


# In[61]:


roi_signal_suite2p.shape


# In[ ]:


get_ipython().run_line_magic('reset_selective', '-f suite2p_data')


# In[ ]:


plt.figure(figsize = (9,3))
plt.plot(tvec,roi_signal_suite2p[0])
plt.axis([0,500,-100,500])


# # Plot Time-series of Selected ROIs

# In[ ]:


# plot suite2p extracted signals
fig, ax = plt.subplots(numROI_2Viz, 1, figsize = (9,10))
for idx, iROI in enumerate(plot_these_roi):
    
    baseline = np.mean(F[iROI])
    
    to_plot = ( (F[iROI]-baseline)/baseline )*100
    to_plot = F[iROI]
    
    ax[idx].plot(tvec, np.transpose( to_plot ), color = colors_roi[idx] );
    ax[idx].axis([0,500,-100,500])
    
    if idx == np.ceil(numROI_2Viz/2-1):
        ax[idx].set_ylabel('Fluorescence Level',fontsize = 20)
        
ax[idx].set_xlabel('Time [s]',fontsize = 20);
plt.savefig(fig_save_dir + 'roi_ts.jpg')
plt.savefig(fig_save_dir + 'roi_ts.eps', format='eps')


# # SIMA 

# In[26]:


# load video data
# open h5 to read, find data key, grab data, then close
h5 = h5py.File(sima_h5_path,'r')
sima_data = np.squeeze(np.array( h5[list(h5.keys())[0]] )).astype('int16') # np.array loads all data into memory
h5.close()


# In[59]:


proj_manual = {'mean_img': np.mean(sima_data, axis = 0), 
               'max_img': np.max(sima_data, axis = 0), 
               'std_img': np.std(sima_data, axis = 0) }


# In[28]:


manual_data_dims = sima_data.shape


# In[29]:


# grab ROI masks from sima (these are probably manually drawn ROIs from imagej)
sima_mask_path = '{}{}_sima_masks.npy'.format(root_dir, filename)

sima_masks = np.load(sima_mask_path)
numROI_sima = sima_masks.shape[0]


# In[30]:


roi_signal_sima = np.empty([numROI_sima,manual_data_dims[0]])
zero_template_manual = np.zeros([manual_data_dims[1], manual_data_dims[2]])
roi_label_loc_manual = []

for iROI in range(numROI_sima):
    
    # calc signal mean across roi pixels
    ypix_roi, xpix_roi = np.where(sima_masks[iROI,:,:] == 1)
    roi_signal_sima[iROI,:] = np.mean(sima_data[:, ypix_roi, xpix_roi  ], 
                         axis = 1)
    
    # make binary map of ROI pixels
    zero_template_manual[ ypix_roi, xpix_roi ] = 1*(iROI+1)
    
    roi_label_loc_manual.append( [np.min(ypix_roi), np.min(xpix_roi)] )
    
roi_signal_sima.shape


# In[129]:


# plot contours and cell numbers
clims = [ np.min(proj_manual['mean_img']), 
        np.max(proj_manual['mean_img'] ) ]

fig, ax = plt.subplots(1, 1, figsize = (12,12))
ax.imshow(proj_manual['mean_img'], cmap = 'gray', vmin = clims[0]*0.8, vmax = clims[1]*0.6)
ax.axis('off')
ax.contour(zero_template_manual, colors = colors_roi);

# plot ROI number
for iROI in range(numROI_sima): 
    ax.text(roi_label_loc_manual[iROI][1], roi_label_loc_manual[iROI][0],  str(iROI), fontsize=15, color = 'white');

plt.title('Manual ROIs', fontsize = 20)
#plt.savefig(fig_save_dir + 'roi_contour_map.jpg')
#plt.savefig(fig_save_dir + 'roi_contour_map.eps', format='eps')


# # Match cells and correlate time-series

# In[132]:


# load csv with cell matching between manual and suite2p
data = pd.read_csv( 'C:\\2pData\\Ivan\\cell_matching.csv', usecols=[0,1]) 
manual_suite2p_cell_links = data[data["Suite2p"].notnull()]
manual_suite2p_cell_links.head() # note that unmatched cells will not show up


# In[133]:


manual_suite2p_cell_links["signal_corr"] = ""


# In[134]:


suite2p_roi = 10 #0 #14 # Vijay: 53 # 51 #11 # 42
manual_roi = 52 #10 #39 # Vijay: 44# 16 #8 # 39

# normalize traces
suite2p_trace = roi_signal_suite2p[suite2p_roi,:]
suite2p_trace = (suite2p_trace - np.mean(suite2p_trace))/np.std(suite2p_trace)
manual_trace = roi_signal_sima[manual_roi,:]
manual_trace = (manual_trace - np.mean(manual_trace))/np.std(manual_trace)

plt.figure(figsize = (7,3))
plt.plot(tvec, suite2p_trace)
plt.plot(tvec, manual_trace, alpha = 0.7)
plt.xlim([0, 600])
roi_corr_coef = np.corrcoef( roi_signal_suite2p[suite2p_roi,:], roi_signal_sima[manual_roi,:] )[0,1]
plt.title('ROI {} Corr Coeff = {:.2f}'.format(suite2p_roi, roi_corr_coef), fontsize = 20)
plt.xlabel('Time [s]', fontsize = 20)
plt.ylabel('Norm Fluorescence', fontsize = 20)
plt.legend(['suite2p', 'manual'], fontsize = 13);


# In[135]:


for index, row in manual_suite2p_cell_links.iterrows():

    manual_roi_id = int(row['Manual'])
    suite2p_roi_id = int(row['Suite2p'])
    #print(manual_roi_id,suite2p_roi_id)
    #print(np.corrcoef( roi_signal_suite2p[suite2p_roi_id,:], roi_signal_sima[manual_roi_id,:] )[0,1])
    
    manual_suite2p_cell_links.loc[index, 'signal_corr'] = np.corrcoef( roi_signal_suite2p[suite2p_roi_id,:], 
                                            roi_signal_sima[manual_roi_id,:] )[0,1]


# In[136]:


mean_corr = manual_suite2p_cell_links['signal_corr'].mean()
print('Mean correlation across ROIs:' + str(mean_corr))


# # Plot suite2p mean image and contours with correl values

# In[137]:


# plot contours and cell numbers
fig, ax = plt.subplots(1, 1, figsize = (12,12))
ax.imshow(hf['std_img'], cmap = 'gray')
ax.axis('off')
ax.contour(zero_template_suite2p, colors = colors_roi);

for iROI in range(num_cells_suite2p):
    
    try:
        roi_corr = manual_suite2p_cell_links.loc[
            manual_suite2p_cell_links['Suite2p'] == float(iROI)]['signal_corr'].iloc[0]
    
        ax.text(roi_label_loc[iROI][1], roi_label_loc[iROI][0],  "{:.2f}".format(roi_corr), fontsize=13, color = 'white')
    except:
        pass
plt.title('Manual-Suite2p Signal Correlation', fontsize = 20)
    #plt.savefig(fig_save_dir + 'roi_contour_map.jpg')
#plt.savefig(fig_save_dir + 'roi_contour_map.eps', format='eps')


# In[138]:


plt.figure()
manual_suite2p_cell_links['signal_corr'].hist(bins=20)
plt.title('Manual-Suite2p Correlation Distribution', fontsize = 20)
plt.xlabel('Pearson Correlation', fontsize = 20)
plt.ylabel('Counts', fontsize = 20)
plt.grid(False)


# In[ ]:




