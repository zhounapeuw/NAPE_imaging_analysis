#!/usr/bin/env python
# coding: utf-8

# # Code to evaluate and compare 2p imaging motion correction
# 
# author: Zhe Charles Zhou (UW NAPE Center)
# 
# Loads in raw and motion corrected data then computes:
# 
# - mean images and zoomed mean image
# - displacement across frames
# - correlation to mean (CM) metric for all datasets
# - Crispness metric
# 
# CM and crispness metric based on methods described in: 
# 
# Pnevmatikakis EA, Giovannucci A. NoRMCorre: An online algorithm for piecewise rigid motion correction of calcium imaging data. J Neurosci Methods. 2017;291:83–94. doi:10.1016/j.jneumeth.2017.07.031
# 
# pre-req input data:
# 
# - raw imaging data in form of h5 or tiff (residing in root folder)
# - motion corrected data (from SIMA, suite2p, and caiman) in form of h5 or tiff (residing in root folder)
# - processed displacement file (from each analysis package) (residing in root folder\displacements\ )

# In[1]:


import tifffile as tiff
import h5py
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

from collections import defaultdict


# In[23]:


# User needs to define the root folder and base file name here

#root_filename = 'VJ_OFCVTA_7_260_D6'
#root_filename = 'itp_lhganiii_bl3_935'
root_filename = '091618 155a day 2 tiffs'

#folder = 'C:\\2pData\\Vijay data\\VJ_OFCVTA_7_D8_trained\\'
#folder = 'C:\\2pData\\Ivan\\itp_lhganiii_bl3_678\\'
folder = 'C:\\2pData\\Christian data\\Same FOV\\Individual Trials\\091618 155a day 2 tiffs\\processed\\'

fps = 5 # USER DEFINE


# In[3]:


# make a dict with entries for each data/motion-correction type

dat_type_names = ['raw','sima','suite2p','caiman']
dat_ext = ['','_sima_mc','_suite2p_mc','_fullcaiman_mc']

tree = lambda: defaultdict(tree) # dictionary: unordered, multi-layered variable storage
dat_dict = tree()

for dat_type, file_ext in zip(dat_type_names, dat_ext): 
    dat_dict[dat_type]['dir'] = os.path.join( folder, '{}{}.h5'.format(root_filename,file_ext) )

dat_dict


# In[4]:


# add and process displacements 
# need to run for sima: /Python/Charles/Vijay_Pipeline.ipynb
# for suite2p: /Python/Charles/suite2p_save_projections&displacements.ipynb
# for caiman: /Documents/GitHub/CaImAn/demos/notebooks/caiman_mc_singleBlock.ipynb

for dat_type in dat_type_names[1:]:

    disp_fpath = '{}displacements\\displacements_{}.npy'.format(folder, dat_type)
    displacement = np.load(disp_fpath)
    tvec = np.linspace(0,len(displacement)/fps,len(displacement))
    plt.plot(tvec,displacement, alpha=0.5)

plt.xlabel('Time [s]',fontsize = 20)
plt.ylabel('Displacement [pixels]',fontsize = 20)
plt.legend(dat_type_names[1:]);


# In[5]:


# function to load tiff data and get data shape
def read_shape_tiff(data_path):
    
    data = tiff.imread(data_path).astype('int16')
    data_shape = data.shape

    return data, data_shape

def read_shape_h5(data_path):
    
    # open h5 to read, find data key, grab data, then close
    h5 = h5py.File(data_path,'r')
    data = np.squeeze(np.array( h5[h5.keys()[0]] )).astype('int16') # np.array loads all data into memory
    h5.close()
    
    data_shape = data.shape
    
    return data, data_shape


# In[6]:


# loop through keys of dictionary, and load video data
for key in dat_dict:

    dat_dict[key]['raw_dat'], dat_dict[key]['dat_dim'] = read_shape_h5(dat_dict[key]['dir'])
    
    # needed b/c suite2p divides intensity values by 2
    if key == 'suite2p':
        dat_dict[key]['raw_dat'] = dat_dict[key]['raw_dat'] * 2
    
    print("{} {}".format(key, dat_dict[key]['dat_dim']))


# In[24]:


""" calculate minimum and max FOVs after motion correction to crop all data to similar dimensions (to facilitate correlation)
 some algorithms crop b/c may be some edge pixels that contain little information due to shifting out of view
# use list comprehension to extract corresponding dimension from each key in the dict """

# FIRST List comprehension
min_ypix = np.min([dat_dict[key]['dat_dim'][1] for key in dat_dict])
min_xpix = np.min([dat_dict[key]['dat_dim'][2] for key in dat_dict])


# # Plot mean images for each analysis dataset

# In[25]:


# function to crop frames equally on each side; measure out from the center
def crop_center(img,cropx,cropy):
    z,y,x = img.shape
    startx = x//2-(cropx//2) # // is floor division
    starty = y//2-(cropy//2)    
    return img[:,starty:starty+cropy,startx:startx+cropx]


# In[26]:


""" use function to crop videos; important for easier aligned comparison of mean imgs, 
but also removing suite2p edge artifacts """

for key in dat_dict: 
    
    # crop data
    dat_dict[key]['raw_dat'] = crop_center(dat_dict[key]['raw_dat'],min_xpix,min_ypix)

    # compute mean image    
    dat_dict[key]['mean_img'] = np.mean(dat_dict[key]['raw_dat'], axis=0)


# In[27]:


# set color intensity limits based on min and max of all data
clims = [ np.min([dat_dict[key]['mean_img'] for key in dat_dict]), 
        np.max([dat_dict[key]['mean_img'] for key in dat_dict])-100 ]
clims


# In[11]:


# function that takes in mean image and plots 
def subplot_mean_img(axs, data_name, mean_img, clims, zoom_window=None):

    im = axs.imshow(mean_img, cmap='gray')
    axs.set_title(data_name, fontsize = 20)
    
    im.set_clim(vmin=clims[0], vmax=clims[1])
    
    if zoom_window is not None:
        
        axs.set_title(data_name + ' Zoom', fontsize = 20)
        axs.axis(zoom_window)
        axs.invert_yaxis()
    axs.axis('off')


# In[28]:


# plot mean images

zoom_window = [200,300,150,250] # [xmin, xmax, ymin, ymax]; LH [150,250,250,350]

fig, axs = plt.subplots(2, 4, figsize=(15, 10))

# FIRST ENUMERATE
# enumerate allows for looping through iterable and provides a count
for idx, key in enumerate(dat_dict): 
    
    subplot_mean_img(axs[0,idx], key, dat_dict[key]['mean_img'], clims)
    
    subplot_mean_img(axs[1,idx], key, dat_dict[key]['mean_img'], clims, zoom_window)


# In[13]:


# plot mean images

fig, axs = plt.subplots(2, 4, figsize=(15, 10))

im0 = axs[0,0].imshow(dat_dict['raw']['mean_img'], cmap='gray')
axs[0,0].set_title('Raw', fontsize = 20)

im1 = axs[0,1].imshow(dat_dict['sima']['mean_img'], cmap='gray')
axs[0,1].set_title('SIMA Corrected', fontsize = 20)

im2 = axs[0,2].imshow(dat_dict['suite2p']['mean_img'], cmap='gray')
axs[0,2].set_title('Suite2p Corrected', fontsize = 20)

im3 = axs[0,3].imshow(dat_dict['caiman']['mean_img'], cmap='gray')
axs[0,3].set_title('Caiman Corrected', fontsize = 20)

im0.set_clim(vmin=clims[0], vmax=clims[1]); im1.set_clim(vmin=clims[0], vmax=clims[1]); 
im2.set_clim(vmin=clims[0], vmax=clims[1]); im3.set_clim(vmin=clims[0], vmax=clims[1])

im3 = axs[1,0].imshow(dat_dict['raw']['mean_img'], cmap='gray')
axs[1,0].set_title('Raw Zoom', fontsize = 20)
axs[1,0].axis(zoom_window)
axs[1,0].invert_yaxis()

im4 = axs[1,1].imshow(dat_dict['sima']['mean_img'], cmap='gray')
axs[1,1].set_title('SIMA Zoom', fontsize = 20)
axs[1,1].axis(zoom_window)
axs[1,1].invert_yaxis()

im5 = axs[1,2].imshow(dat_dict['suite2p']['mean_img'], cmap='gray')
axs[1,2].set_title('Suite2p Zoom', fontsize = 20)
axs[1,2].axis(zoom_window)
axs[1,2].invert_yaxis()

im6 = axs[1,3].imshow(dat_dict['caiman']['mean_img'], cmap='gray')
axs[1,3].set_title('Caiman Zoom', fontsize = 20)
axs[1,3].axis(zoom_window)
axs[1,3].invert_yaxis()

im3.set_clim(vmin=clims[0], vmax=clims[1]); im4.set_clim(vmin=clims[0], vmax=clims[1]); 
im5.set_clim(vmin=clims[0], vmax=clims[1]); im6.set_clim(vmin=clims[0], vmax=clims[1])


# # Compute Frame-by-frame correlation to the mean image

# In[14]:


# function to compute frame-resolved correlation to reference mean image
def corr2_all_frames(data,ref):
    cor_all = np.empty([data.shape[0],])
    
    for iframe,frame in enumerate(data):
        print 'frame {0}\r'.format(iframe),
        # pearson corr used in NoRMCorre paper
        cor_all[iframe] = np.corrcoef(np.ndarray.flatten(frame), np.ndarray.flatten(ref))[0,1] # 
        
    return cor_all


# In[15]:


for key in dat_dict: 
    
    print('Corr {} Data'.format(key))
    
    # we'll correlate each frame within a dataset to the mean image of that dataset
    dat_dict[key]['frame_corr'] = corr2_all_frames(dat_dict[key]['raw_dat'],dat_dict[key]['mean_img'])


# In[16]:


# plot correlation as function of time 
fig, ax = plt.subplots(1, 1, figsize=(10,5), sharey=True)

num_samples = dat_dict['raw']['dat_dim'][0]
tvec = np.linspace(0,num_samples/fps,num_samples)

for key in dat_dict: 

    plt.plot(tvec,dat_dict[key]['frame_corr'], alpha = 0.7)

plt.xlabel('Time [s]', fontsize=20)
plt.ylabel('Pearson Correlation', fontsize=20)
plt.legend(dat_type_names);


# In[17]:


# calculate correlation means for bar graph
corr_means = [ np.mean(dat_dict[key]['frame_corr']) for key in dat_dict ]
display(corr_means)

# calculate SEMs
corr_sems = [np.std(dat_dict[key]['frame_corr'])/math.sqrt(len(dat_dict[key]['frame_corr']))
             for key in dat_dict]

display(corr_sems)


# In[18]:


x_pos = np.arange(len(dat_type_names)) # find x tick locations for replacement with condition names

fig, ax = plt.subplots()
ax.bar(x_pos, corr_means, yerr=corr_sems, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylim([ np.min(corr_means)-0.01, np.max(corr_means)+0.01 ])
ax.set_xticks(x_pos)
ax.set_xticklabels(dat_type_names, fontsize = 20)
ax.set_ylabel('Pearson Correlation', fontsize = 20);


# # Calculate Crispness
# 
# https://www.sciencedirect.com/science/article/pii/S0165027017302753#tbl0005
# 
# \begin{equation*}
# C(I)   = \lVert \lvert  \nabla I \rvert \rVert_F
# \end{equation*}
# 
# where
# 
# \begin{equation*} C(I) \end{equation*}
# 
# is the crispness value for image I
# 
# \begin{equation*} \nabla I \end{equation*}
# 
# is the gradient of image I (np.gradient gives x and y directions for each pixel's vector)
# 
# \begin{equation*} \lvert \rvert_F \end{equation*}
# 
# is the pixel-wise magnitude
# 
# \begin{equation*} \lVert \rVert_F \end{equation*}
# 
# is the frobenius norm (formally square root of the absolute sum of squares across matrix elements , but can be thought of as a way to summarize the magnitudes across pixels)
# 
# 
# 
# 

# In[19]:


# calculate gradient vector field; https://stackoverflow.com/questions/30079740/image-gradient-vector-field-in-python
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter

zoom_window = [150,250,200,300]

I = np.flipud(dat_dict['suite2p']['mean_img'])
p = np.asarray(I)
w,h = I.shape
complex_y = complex(0,w)
complex_x = complex(0,h)
y, x = np.mgrid[0:h:complex_y, 0:w:complex_x] # CZ: end dimensions need to match input; 
# complex b/c gradient output has vector angle and amplitude info

dy, dx = np.gradient(p) # for each pixel, calculate gradient vectors (dir and mag of largest change in pixel intensity)
skip = (slice(None, None, 3), slice(None, None, 3)) # skip a few pixels for better visualization

fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(np.flipud(I), extent=[x.min(), x.max(), y.min(), y.max()]) # show original img
ax.quiver(x[skip], y[skip], dx[skip], dy[skip]) # plot vectors

ax.set(aspect=1, title='Quiver Plot')
ax.set_title('Quiver Plot', fontsize = 30)
ax.axis(zoom_window)
plt.axis('off')
plt.show()


# In[20]:


# calculate entry(pixel)-wise magnitude

def calc_all_vect_mag(dy,dx):
    
    # initialize pixel-wise mag array
    h_pix = dy.shape[0]
    w_pix = dy.shape[1]
    all_vect_mag = np.empty( [h_pix,w_pix] )
    
    # np.gradient gives x/y vector components; need to calculate composite magnitude for each pixel
    # np.ndenumerate returns 2d index for each entry
    for index, x in np.ndenumerate(dy):
    
        ycoord = index[0] 
        xcoord = index[1]
        
        all_vect_mag[ycoord,xcoord] = (dx[ycoord,xcoord] ** 2 + dy[ycoord,xcoord] ** 2) ** 0.5
    
    return all_vect_mag


# In[21]:


for key in dat_dict: 
    
    # first calculate gradient (vector for each pixel)
    img_in = np.asarray(np.flipud(dat_dict[key]['mean_img']))
    dy, dx = np.gradient(img_in)

    dat_dict[key]['grad_mag'] = calc_all_vect_mag(dy,dx)


# In[22]:


# calculate Frobenius norm 
print 'raw Crispness: ' , np.linalg.norm(dat_dict['raw']['grad_mag'], ord = 'fro')
print 'sima Crispness: ' , np.linalg.norm(dat_dict['sima']['grad_mag'], ord = 'fro')
print 'suite2p Crispness: ' , np.linalg.norm(dat_dict['suite2p']['grad_mag'], ord = 'fro')
print 'caiman Crispness: ' , np.linalg.norm(dat_dict['caiman']['grad_mag'], ord = 'fro')


# # Calculate Optical Flow

# In[ ]:


import cv2
import logging


# In[ ]:


pyr_scale=.5
levels=3
winsize=100
iterations=15
poly_n=5
poly_sigma=1.2 / 5
flags=0
play_flow=False
resize_fact_flow=.2
template=None


# In[ ]:


key = 'suite2p'
tmpl = dat_dict[key]['mean_img']


# In[ ]:


norms = []
flows = []
count = 0

for fr in dat_dict[key]['raw_dat']:
    if count % 100 == 0:
        logging.debug(count)

    count += 1
    flow = cv2.calcOpticalFlowFarneback(
        tmpl, fr, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)

    if play_flow:
        pl.subplot(1, 3, 1)
        pl.cla()
        pl.imshow(fr, vmin=0, vmax=300, cmap='gray')
        pl.title('movie')
        pl.subplot(1, 3, 3)
        pl.cla()
        pl.imshow(flow[:, :, 1], vmin=vmin, vmax=vmax)
        pl.title('y_flow')

        pl.subplot(1, 3, 2)
        pl.cla()
        pl.imshow(flow[:, :, 0], vmin=vmin, vmax=vmax)
        pl.title('x_flow')
        pl.pause(.05)

    n = np.linalg.norm(flow)
    flows.append(flow)
    norms.append(n)


# In[ ]:


plt.plot(norms)


# # Perform KLT Tracking with OpenCV
# 
# Based on: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
# 
# Also informative: https://stackoverflow.com/questions/18863560/how-does-klt-work-in-opencv
# 
# https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

# In[ ]:


# grab reference frame
ref_frame = raw_dat[0,:,:]
this_frame = raw_dat[100,:,:]


# In[ ]:


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


# In[ ]:


p0 = cv.goodFeaturesToTrack(ref_frame, mask = None, **feature_params)
p0.shape


# In[ ]:


# Create some random colors
color = np.random.randint(0,255,(100,3))
# Create a mask image for drawing purposes
mask = np.zeros_like(ref_frame)
frame_idx = 1

while(1):
    
    this_frame = raw_dat[frame_idx,:,:]
    
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(ref_frame, this_frame, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(this_frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(this_frame,mask)
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27 or frame_idx == raw_dat_dim[0]-1:
        break
    # Now update the previous frame and previous points
    old_gray = this_frame.copy()
    p0 = good_new.reshape(-1,1,2)
    
    frame_idx += 1



# In[ ]:




