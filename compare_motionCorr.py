#!/usr/bin/env python
# coding: utf-8

# # Code to evaluate and compare 2p imaging motion correction
# 
# author: Zhe Charles Zhou (UW NAPE Center)
# 
# Loads in raw and motion corrected data then computes:
# 
# - correlation to mean (CM) metric for all datasets
# - Crispness metric
# 
# pre-req input data:
# 
# - raw imaging data in form of h5
# - motion corrected data in form of h5

# In[2]:


import tifffile as tiff
import h5py
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

from collections import defaultdict


# In[3]:


root_filename = 'VJ_OFCVTA_7_260_D6'
root_filename = 'itp_lhganiii_bl3_935'
#root_filename = '091618 155a day 2 tiffs'

folder = 'C:\\2pData\\Vijay data\\VJ_OFCVTA_7_D8_trained\\'
folder = 'C:\\2pData\\Ivan\\itp_lhganiii_bl3_678\\'
#folder = 'C:\\2pData\\Christian data\\Same FOV\\Individual Trials\\091618 155a day 2 tiffs\\processed\\'

fps = 5


# In[4]:


# make a dict with entries for each data/motion-correction type

dat_type_names = ['raw','sima','suite2p','caiman']
dat_ext = ['','_sima_mc','_suite2p_mc','_fullcaiman_mc']

tree = lambda: defaultdict(tree)
dat_dict = tree()

for idx,dat_type in enumerate(dat_type_names):
    dat_dict[dat_type]['dir'] = os.path.join( folder, '{}{}.h5'.format(root_filename,dat_ext[idx]) )

dat_dict


# In[5]:


# function to load tiff data and get data shape
def read_shape_tiff(data_path):
    
    data = tiff.imread(data_path).astype('int16')
    data_shape = data.shape
    
    print("{} {}".format(data.dtype, data.shape))
    
    return data, data_shape

def read_shape_h5(data_path):
    
    # open h5 to read, find data key, grab data, then close
    h5 = h5py.File(data_path,'r')
    data = np.squeeze(np.array( h5[h5.keys()[0]] )).astype('int16') # np.array loads all data into memory
    h5.close()
    
    data_shape = data.shape
    
    print("{} {}".format(data.dtype, data.shape))
    
    return data, data_shape


# In[6]:


for key in dat_dict:

    dat_dict[key]['raw_dat'], dat_dict[key]['dat_dim'] = read_shape_h5(dat_dict[key]['dir'])
    
    # needed b/c suite2p divides intensity values by 2
    if key == 'suite2p':
        dat_dict[key]['raw_dat'] = dat_dict[key]['raw_dat'] * 2


# In[47]:


num_samples = dat_dict['raw']['dat_dim'][0]


# In[7]:


# calculate minimum and max FOVs after motion correction to crop all data to similar dimensions (to facilitate correlation)
# use list comprehension to extract corresponding dimension from each key in the dict
min_ypix = np.min([dat_dict[key]['dat_dim'][1] for key in dat_dict])
min_xpix = np.min([dat_dict[key]['dat_dim'][2] for key in dat_dict])


# # Perform correlation to mean image

# In[8]:


# function to crop frames equally on each side
def crop_center(img,cropx,cropy):
    z,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[:,starty:starty+cropy,startx:startx+cropx]


# In[9]:


""" use function to crop videos; important for easier aligned comparison of mean imgs, 
but also removing suite2p edge artifacts """

for key in dat_dict: 
    
    # crop data
    dat_dict[key]['raw_dat'] = crop_center(dat_dict[key]['raw_dat'],min_xpix,min_ypix)

    # compute mean image    
    dat_dict[key]['mean_img'] = np.mean(dat_dict[key]['raw_dat'], axis=0)


# In[17]:


# set color intensity limits based on min and max of all data
clims = [ np.min([dat_dict[key]['mean_img'] for key in dat_dict]), 
        np.max([dat_dict[key]['mean_img'] for key in dat_dict])-100 ]


# In[38]:


def subplot_mean_img(axs, data_name, mean_img, clims, zoom_window=None):

    im = axs.imshow(mean_img, cmap='gray')
    axs.set_title(data_name, fontsize = 20)
    
    im.set_clim(vmin=clims[0], vmax=clims[1])
    
    if zoom_window is not None:
        
        axs.set_title(data_name + ' Zoom', fontsize = 20)
        axs.axis(zoom_window)
        axs.invert_yaxis()


# In[39]:


# plot mean images

fig, axs = plt.subplots(2, 4, figsize=(15, 10))

for idx, key in enumerate(dat_dict): 
    
    subplot_mean_img(axs[0,idx], key, dat_dict[key]['mean_img'], clims)
    
    subplot_mean_img(axs[1,idx], key, dat_dict[key]['mean_img'], clims, zoom_window)


# In[40]:


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

im0.set_clim(vmin=clim[0], vmax=clim[1]); im1.set_clim(vmin=clim[0], vmax=clim[1]); 
im2.set_clim(vmin=clim[0], vmax=clim[1]); im3.set_clim(vmin=clim[0], vmax=clim[1])

zoom_window = [150,250,200,300] # [xmin, xmax, ymin, ymax]

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

im3.set_clim(vmin=clim[0], vmax=clim[1]); im4.set_clim(vmin=clim[0], vmax=clim[1]); 
im5.set_clim(vmin=clim[0], vmax=clim[1]); im6.set_clim(vmin=clim[0], vmax=clim[1])

#fig.colorbar(im0)



# In[42]:


# function to compute frame-resolved correlation to reference mean image
def corr2_all_frames(data,ref):
    cor_all = np.empty([data.shape[0],])
    
    for iframe,frame in enumerate(data):
        print 'frame {0}\r'.format(iframe),
        cor_all[iframe] = np.corrcoef(np.ndarray.flatten(frame), np.ndarray.flatten(ref))[0,1] #  corr2(np.ndarray.flatten(frame), np.ndarray.flatten(ref)) # 
        
    return cor_all


# In[43]:


for key in dat_dict: 
    
    print('Corr {} Data'.format(key))
    
    dat_dict[key]['frame_corr'] = corr2_all_frames(dat_dict[key]['raw_dat'],dat_dict[key]['mean_img'])


# In[ ]:


# run frame-by-frame correlation to mean image
print('Corr Raw Data')
raw_corr2 = corr2_all_frames(raw_dat,raw_mean)
print('Corr Sima Data')
sima_corr2 = corr2_all_frames(sima_dat,sima_mean)
print('Corr Suite2p Data')
suite2p_corr2 = corr2_all_frames(suite2p_dat,suite2p_mean)
print('Corr Caiman Data')
caiman_corr2 = corr2_all_frames(caiman_dat,caiman_mean)


# In[48]:


# plot correlation as function of time 
fig, ax = plt.subplots(1, 1, figsize=(10,5), sharey=True)

tvec = np.linspace(0,num_samples/fps,num_samples)

plt.plot(tvec,dat_dict['raw']['frame_corr'])
plt.plot(tvec,dat_dict['sima']['frame_corr'])
plt.plot(tvec,dat_dict['suite2p']['frame_corr'])
plt.plot(tvec,dat_dict['caiman']['frame_corr'])
plt.xlabel('Time [s]', fontsize=20)
plt.ylabel('Pearson Correlation', fontsize=20)
plt.legend(dat_type_names);


# In[49]:


# calculate correlation means
raw_corr_mean = np.mean(dat_dict['raw']['frame_corr'])
sima_corr_mean = np.mean(dat_dict['sima']['frame_corr'])
suite2p_corr_mean = np.mean(dat_dict['suite2p']['frame_corr'])
caiman_corr_mean = np.mean(dat_dict['caiman']['frame_corr'])
corr_means = [raw_corr_mean, sima_corr_mean, suite2p_corr_mean, caiman_corr_mean]
display(corr_means)

# calculate SEMs
raw_corr_sem = np.std(dat_dict['raw']['frame_corr'])/math.sqrt(len(dat_dict['raw']['frame_corr']))
sima_corr_sem = np.std(dat_dict['sima']['frame_corr'])/math.sqrt(len(dat_dict['sima']['frame_corr']))
suite2p_corr_sem = np.std(dat_dict['suite2p']['frame_corr'])/math.sqrt(len(dat_dict['suite2p']['frame_corr']))
caiman_corr_sem = np.std(dat_dict['caiman']['frame_corr'])/math.sqrt(len(dat_dict['caiman']['frame_corr']))
corr_sems = [raw_corr_sem, sima_corr_sem, suite2p_corr_sem, caiman_corr_sem]
display(corr_sems)


# In[50]:


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

# In[53]:


# calculate gradient vector field; https://stackoverflow.com/questions/30079740/image-gradient-vector-field-in-python
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter

I = np.flipud(dat_dict['sima']['mean_img'])
p = np.asarray(I)
w,h = I.shape
complex_y = complex(0,dat_dict['sima']['dat_dim'][1])
complex_x = complex(0,dat_dict['sima']['dat_dim'][2])
y, x = np.mgrid[0:h:complex_y, 0:w:complex_x] # CZ: end dimensions need to match input

dy, dx = np.gradient(p)
skip = (slice(None, None, 3), slice(None, None, 3))

fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(np.flipud(I), extent=[x.min(), x.max(), y.min(), y.max()]) # show original img
ax.quiver(x[skip], y[skip], dx[skip], dy[skip]) # plot vectors

ax.set(aspect=1, title='Quiver Plot')
ax.set_title('Quiver Plot', fontsize = 30)
ax.axis([150,250,150,250])
plt.show()


# In[54]:


# calculate entry-wise magnitude

# CZ: why use class?
# class that takes in gradient x and y vector components and has a method to calculate magnitude
class Vector(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def vector_mag(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


# In[55]:


def calc_all_vect_mag(dy,dx):
    
    h_pix = dy.shape[0]
    w_pix = dy.shape[1]
    
    all_vect_mag = np.empty( [h_pix,w_pix] )
    
    for index, x in np.ndenumerate(dy):
    
        ycoord = index[0] 
        xcoord = index[1]
        comb_vect = Vector(dx[ycoord,xcoord], dy[ycoord,xcoord])
        all_vect_mag[ycoord,xcoord] = comb_vect.vector_mag()
    
    return all_vect_mag


# In[56]:


for key in dat_dict: 
    
    img_in = np.asarray(np.flipud(dat_dict[key]['mean_img']))
    dy, dx = np.gradient(img_in)

    dat_dict[key]['grad_mag'] = calc_all_vect_mag(dy,dx)


# In[57]:


# calculate Frobenius norm
print 'raw Crispness: ' , np.linalg.norm(dat_dict['raw']['grad_mag'], ord = 'fro')
print 'sima Crispness: ' , np.linalg.norm(dat_dict['sima']['grad_mag'], ord = 'fro')
print 'suite2p Crispness: ' , np.linalg.norm(dat_dict['suite2p']['grad_mag'], ord = 'fro')
print 'caiman Crispness: ' , np.linalg.norm(dat_dict['caiman']['grad_mag'], ord = 'fro')


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




