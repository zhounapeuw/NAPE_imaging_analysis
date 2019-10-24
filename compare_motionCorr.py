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
# - raw imaging data in form of tif
# - motion corrected data in form of tif

# In[207]:


import tifffile as tiff
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

from collections import defaultdict


# In[286]:


root_filename = 'temp'
filename_sima = root_filename + '_mc'
filename_suite2p = root_filename + '_suite2p_mc'

#folder = 'C:/2pData/Vijay data/VJ_OFC_6_D9_trained/'
folder = 'C:\\2pData\\Vijay data\\VJ_OFCVTA_7_D8_trained\\cutTiff\\'


fps = 5


# In[287]:


# make a dict with entries for each data/motion-correction type

dat_type_names = ['raw','sima','suite2p']
dat_ext = ['','_sima_mc','_suite2p_mc']

tree = lambda: defaultdict(tree)
dat_dict = tree()

for idx,dat_type in enumerate(dat_type_names):
    dat_dict[dat_type]['dir'] = os.path.join( folder, 'temp{}.tif'.format(dat_ext[idx]) )

dat_dict


# In[288]:


# function to load tiff data and get data shape
def read_shape_tiff(data_path):
    
    data = tiff.imread(data_path).astype('uint8')
    data_shape = data.shape
    
    print("{} {}".format(data.dtype, data.shape))
    
    return data, data_shape


# In[289]:


# load data 
raw_dat, raw_dat_dim = read_shape_tiff(dat_dict['raw']['dir'])
sima_dat, sima_dat_dim = read_shape_tiff(dat_dict['sima']['dir'])
suite2p_dat, suite2p_dim = read_shape_tiff(dat_dict['suite2p']['dir'])
suite2p_dat = suite2p_dat * 2 # needed b/c suite2p divides intensity values by 2


# In[290]:


# calculate minimum and max FOVs after motion correction to crop all data to similar dimensions (to facilitate correlation)
min_ypix = np.min([raw_dat_dim[1], sima_dat_dim[1], suite2p_dim[1]])
min_xpix = np.min([raw_dat_dim[2], sima_dat_dim[2], suite2p_dim[2]])


# In[291]:


# function to crop frames equally on each side
def crop_center(img,cropx,cropy):
    z,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[:,starty:starty+cropy,startx:startx+cropx]


# In[292]:


raw_dat = crop_center(raw_dat,min_xpix,min_ypix)
suite2p_dat = crop_center(suite2p_dat,min_xpix,min_ypix)


# # Perform correlation to mean image

# In[322]:


# calculate mean image
raw_mean = np.mean(raw_dat, axis=0)
sima_mean = np.mean(sima_dat, axis=0)
suite2p_mean = np.mean(suite2p_dat, axis=0)


# In[366]:


# plot mean images

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

im0 = axs[0,0].imshow(raw_mean, cmap='gray')
axs[0,0].set_title('Raw', fontsize = 20)

im1 = axs[0,1].imshow(sima_mean, cmap='gray')
axs[0,1].set_title('SIMA Corrected', fontsize = 20)

im2 = axs[0,2].imshow(suite2p_mean, cmap='gray')
axs[0,2].set_title('Suite2p Corrected', fontsize = 20)

im0.set_clim(vmin=15, vmax=165); im1.set_clim(vmin=15, vmax=165); im2.set_clim(vmin=15, vmax=165)

im3 = axs[1,0].imshow(raw_mean, cmap='gray')
axs[1,0].set_title('Raw Zoom', fontsize = 20)
axs[1,0].axis([150,250,250,350])

im4 = axs[1,1].imshow(sima_mean, cmap='gray')
axs[1,1].set_title('SIMA Zoom', fontsize = 20)
axs[1,1].axis([150,250,250,350])

im5 = axs[1,2].imshow(suite2p_mean, cmap='gray')
axs[1,2].set_title('Suite2p Zoom', fontsize = 20)
axs[1,2].axis([150,250,250,350])

im3.set_clim(vmin=30, vmax=180); im4.set_clim(vmin=30, vmax=180); im5.set_clim(vmin=30, vmax=180)

#fig.colorbar(im0)



# In[295]:


# 2 functions for calculating 2d correlation
def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r

# function to compute frame-resolved correlation to reference mean image
def corr2_all_frames(data,ref):
    cor_all = np.empty([data.shape[0],])
    
    for iframe,frame in enumerate(data):
        print 'frame {0}\r'.format(iframe),
        cor_all[iframe] = np.corrcoef(np.ndarray.flatten(frame), np.ndarray.flatten(ref))[0,1] #  corr2(np.ndarray.flatten(frame), np.ndarray.flatten(ref)) # 
        
    return cor_all


# In[315]:


# run frame-by-frame correlation to mean image
raw_corr2 = corr2_all_frames(raw_dat,raw_mean)
sima_corr2 = corr2_all_frames(sima_dat,sima_mean)
suite2p_corr2 = corr2_all_frames(suite2p_dat,suite2p_mean)


# In[316]:


# plot correlation as function of time 
fig, ax = plt.subplots(1, 1, figsize=(10,5), sharey=True)

tvec = np.linspace(0,raw_dat_dim[0]/fps,raw_dat_dim[0])

plt.plot(tvec,raw_corr2)
plt.plot(tvec,sima_corr2)
plt.plot(tvec,suite2p_corr2)
plt.xlabel('Time [s]', fontsize=20)
plt.ylabel('Pearson Correlation', fontsize=20)
plt.legend(dat_type_names);


# In[317]:



x_pos = np.arange(len(dat_type_names)) # 


# In[318]:


# calculate correlation means
raw_corr_mean = np.mean(raw_corr2)
sima_corr_mean = np.mean(sima_corr2)
suite2p_corr_mean = np.mean(suite2p_corr2)
corr_means = [raw_corr_mean, moCorr_corr_mean, suite2p_corr_mean]
display(corr_means)

# calculate SEMs
raw_corr_sem = np.std(raw_corr2)/math.sqrt(len(raw_corr2))
sima_corr_sem = np.std(sima_corr2)/math.sqrt(len(sima_corr2))
suite2p_corr_sem = np.std(suite2p_corr2)/math.sqrt(len(suite2p_corr2))
corr_sems = [raw_corr_sem, sima_corr_sem, suite2p_corr_sem]
display(corr_sems)


# In[319]:


fig, ax = plt.subplots()
ax.bar(x_pos, corr_means, yerr=corr_sems, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylim([ np.min(corr_means)-0.01, np.max(corr_means)+0.01 ])
ax.set_xticks(x_pos)
ax.set_xticklabels(dat_type_names, fontsize = 20)
ax.set_ylabel('Pearson Correlation', fontsize = 20);


# # Calculate Crispness
# 
# https://www.sciencedirect.com/science/article/pii/S0165027017302753#tbl0005

# In[364]:


# calculate gradient vector field; https://stackoverflow.com/questions/30079740/image-gradient-vector-field-in-python
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter

I = np.flipud(moCorr_mean)
p = np.asarray(I)
w,h = I.shape
y, x = np.mgrid[0:h:510j, 0:w:501j] # CZ: end dimensions need to match input

dy, dx = np.gradient(p)
skip = (slice(None, None, 3), slice(None, None, 3))

fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(np.flipud(I), extent=[x.min(), x.max(), y.min(), y.max()]) # show original img
ax.quiver(x[skip], y[skip], dx[skip], dy[skip]) # plot vectors

ax.set(aspect=1, title='Quiver Plot')
ax.set_title('Quiver Plot', fontsize = 30)
ax.axis([150,250,150,250])
plt.show()


# In[307]:


# calculate entry-wise magnitude

# class that takes in gradient x and y vector components and has a method to calculate magnitude
class Vector(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def vector_mag(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


# In[308]:


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


# In[309]:


img_in = np.asarray(np.flipud(raw_mean))
dy, dx = np.gradient(img_in)

raw_grad_mag = calc_all_vect_mag(dy,dx)


# In[310]:


img_in = np.asarray(np.flipud(sima_mean))
dy, dx = np.gradient(img_in)

sima_grad_mag = calc_all_vect_mag(dy,dx)


# In[311]:


img_in = np.asarray(np.flipud(suite2p_mean))
dy, dx = np.gradient(img_in)

suite2p_grad_mag = calc_all_vect_mag(dy,dx)


# In[312]:


# calculate Frobenius norm
print 'raw Crispness: ' , np.linalg.norm(raw_grad_mag, ord = 'fro')
print 'sima Crispness: ' , np.linalg.norm(sima_grad_mag, ord = 'fro')
print 'suite2p Crispness: ' , np.linalg.norm(suite2p_grad_mag, ord = 'fro')


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




