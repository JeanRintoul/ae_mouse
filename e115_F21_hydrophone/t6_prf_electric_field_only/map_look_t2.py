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
sys.path.append('D:\\ae_mouse')  #  so that we can import from the parent folder. 
import mouse_library as m
import pandas as pd 
# matplotlib.rc('xtick', labelsize=14) 
# matplotlib.rc('ytick', labelsize=14) 
fonts = 14
matplotlib.rcParams.update({'font.size': fonts})

interp_method = 'bicubic'
folder  = 'D:\\ae_mouse\\e109_prf_spatial_map\\t2\\'
dfFile  = pd.read_csv(folder + 'prf1020_diff.csv', sep=',')
data    = dfFile.to_numpy()[0:19,1:15]
mm_AP   = data[0:19,0]
a,b     = data.shape 
# print (mm_AP,data,a,b)
# print (data,a,b)


carrierFile     = pd.read_csv(folder + 'prf1020_carrier.csv', sep=',')
carrierdata     = carrierFile.to_numpy()[0:19,1:15]
mm_AP           = carrierdata[0:19,0]
a,b             = carrierdata.shape 
# 

sfFile    = pd.read_csv(folder + 'prf1020_sum.csv', sep=',')
sfdata    = sfFile.to_numpy()[0:19,1:15]
mm_AP     = sfdata[0:19,0]
a,b       = sfdata.shape 
# 
# print (carrier2data)

# print (carrierdata)
extent_params = [-4,5,3.5,-3.5]
# # 
img_df = data.astype('uint8')

fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111)
im  = ax.imshow(data,cmap='Reds',extent=extent_params,alpha=1.0,interpolation=interp_method,vmin=0,vmax = np.max(data) )
# cbar = fig.colorbar(im,fraction=0.046, pad=0.04)
# cbar.set_label('$\mu V$', rotation=90,fontsize=fonts)
# ax.set_title('$PRF \Delta f$ ',fontsize=fonts)
# ax.set_xlabel('mm',fontsize=fonts)
# ax.set_ylabel('mm',fontsize=fonts)
plt.axis('off')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plot_filename = 'phantom_calibration_map_df'
plt.savefig(plot_filename +".png",transparent=True,
           pad_inches=0,bbox_inches='tight')
plt.show()
# 
# 

# 
# carrier_img = carrierdata.astype('uint8')

fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111)
im = ax.imshow(carrierdata,cmap='Reds',extent=extent_params,alpha=1.0,interpolation=interp_method,vmin=0,vmax = np.max(carrierdata) )
# cbar = fig.colorbar(im,fraction=0.046, pad=0.04)
# ax.set_title('$acoustic f@ 500kHz$ ',fontsize=fonts)
# cbar.set_label('$\mu V$', rotation=90,fontsize=fonts)
# ax.set_xlabel('mm',fontsize=fonts)
# ax.set_ylabel('mm',fontsize=fonts)
plt.axis('off')
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plot_filename = 'phantom_calibration_map_carrier'
plt.savefig(plot_filename +".png",transparent=True,
           pad_inches=0,bbox_inches='tight')
plt.show()

# carrier2_img = carrier2data.astype('uint8')

fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111)
im = ax.imshow(sfdata,cmap='Reds',extent=extent_params,alpha=1.0,interpolation=interp_method,vmin=0,vmax = np.max(sfdata) )
# cbar = fig.colorbar(im,fraction=0.046, pad=0.04)
# ax.set_title('$rf voltage f@ 501kHz$ ',fontsize=fonts)
# ax.set_title('$PRF \Sigma f$ ',fontsize=fonts)
# cbar.set_label('$\mu V$', rotation=90,fontsize=fonts)
# ax.set_xlabel('mm',fontsize=fonts)
# ax.set_ylabel('mm',fontsize=fonts)
plt.axis('off')
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plot_filename = 'phantom_calibration_map_sf'
plt.savefig(plot_filename +".png",transparent=True,
           pad_inches=0,bbox_inches='tight')
plt.show()