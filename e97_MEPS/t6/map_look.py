'''

Title: demodulation and a few metrics. Shows FFT comparison of Demodded and original, and time series. 

Author: Jean Rintoul
Date: 02.02.2023

'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.signal import kaiserord, lfilter, firwin, freqz
import pandas as pd
import sys
sys.path.append('D:\\mouse_aeti')  #  so that we can import from the parent folder. 
import mouse_library as m
import pandas as pd 
# matplotlib.rc('xtick', labelsize=14) 
# matplotlib.rc('ytick', labelsize=14) 
fonts = 14
matplotlib.rcParams.update({'font.size': fonts})

interp_method = 'bicubic'
folder  = 'D:\\mouse_aeti\\e97_MEPS\\t6\\'
dfFile  = pd.read_csv('t6_df.csv', sep=',')
data    = dfFile.to_numpy()[0:15,1:16]
# mm_AP   = data[0:12,0]
a,b     = data.shape 
print (data,a,b)
# print (data,a,b)

carrierFile  = pd.read_csv('t6_carrier.csv', sep=',')
carrierdata    = carrierFile.to_numpy()[0:15,1:16]

sfFile  = pd.read_csv('t6_sf.csv', sep=',')
sfdata    = sfFile.to_numpy()[0:15,1:16]

# # 
img = data.astype('uint16')
fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111)
im  = ax.imshow(img,cmap='inferno',extent=[-4,2.5,4.0,-3.5],alpha=1.0,interpolation=interp_method )
cbar = fig.colorbar(im,fraction=0.046, pad=0.04)
cbar.set_label('$\mu V$', rotation=90,fontsize=fonts)
ax.set_title('$\Delta f$ ',fontsize=fonts)
ax.set_xlabel('mm',fontsize=fonts)
ax.set_ylabel('mm',fontsize=fonts)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plot_filename = 'saline_calibration_map_df'
plt.savefig(plot_filename +".png",transparent=True,
           pad_inches=0,bbox_inches='tight')
plt.show()
# # 
# # 
carrier_img = carrierdata.astype('uint16')
fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111)
im = ax.imshow(carrier_img,cmap='inferno',extent=[-4,2.5,4.0,-3.5],alpha=1.0,interpolation=interp_method)
cbar = fig.colorbar(im,fraction=0.046, pad=0.04)
ax.set_title('$f@ 500kHz$ ',fontsize=fonts)
cbar.set_label('$\mu V$', rotation=90,fontsize=fonts)
ax.set_xlabel('mm',fontsize=fonts)
ax.set_ylabel('mm',fontsize=fonts)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plot_filename = 'saline_calibration_map_carrier'
plt.savefig(plot_filename +".png",transparent=True,
           pad_inches=0,bbox_inches='tight')
plt.show()
# # 
# # 
sf_img = sfdata.astype('uint16')
fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111)
im = ax.imshow(sf_img,cmap='inferno',extent=[-4,2.5,4.0,-3.5],alpha=1.0,interpolation=interp_method)
cbar = fig.colorbar(im,fraction=0.046, pad=0.04)
ax.set_title('$\Sigma f$',fontsize=fonts)
cbar.set_label('$\mu V$', rotation=90,fontsize=fonts)
ax.set_xlabel('mm',fontsize=fonts)
ax.set_ylabel('mm',fontsize=fonts)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plot_filename = 'saline_calibration_map_sf'
plt.savefig(plot_filename +".png",transparent=True,
           pad_inches=0,bbox_inches='tight')
plt.show()