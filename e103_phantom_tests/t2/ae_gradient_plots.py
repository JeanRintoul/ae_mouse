'''

Title: 
Author: Jean Rintoul
Date: 27.05.2023

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

fonts = 14
matplotlib.rcParams.update({'font.size': fonts})
# start_files = 1
# end_files   = 6
# files       = np.linspace(start_files,end_files,(end_files-start_files+1),dtype=np.int16)
v_out       = [0.0,0.5,1.0,4,6,12]
df          = [5,34,66,235,367,765]
carrier     = [75,69,56,56,69,39]
# 
#
p_out       = [0,0.01,0.05,0.08,0.1,0.11,0.15]
p_df        = [0.83,33,84,119,128,136,144]
p_carrier   = [6,6,40,66,31,70,59]
pressure    = [0,0.1,0.6,0.9,1.0,1.25,1.6]  # MPa
# 
# 
fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111)
ax.plot(v_out,df,'k')
ax.plot(v_out,carrier,'r')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Voltage output(V)')
ax.set_ylabel('$\mu V$')
plt.legend(['$\Delta f$','$f@ 500kHz$'],loc='upper left',framealpha=0.0)
plot_filename = 'gradient_wrt_voltage'
plt.savefig(plot_filename +".png",transparent=True,
           pad_inches=0,bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111)
# ax.plot(pressure,p_df,'g')
ax.plot(pressure,p_df,'k')
ax.plot(pressure,p_carrier,'r')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Pressure Output (MPa)')
ax.set_ylabel('$\mu V$')
plt.legend(['$\Delta f$','$f@ 500kHz$'],loc='upper left',framealpha=0.0)
plot_filename = 'gradient_wrt_pressure'
plt.savefig(plot_filename +".png",transparent=True,
           pad_inches=0,bbox_inches='tight')
plt.show()

# dfFile  = pd.read_csv('t3_dfc.csv', sep=',')
# data    = dfFile.to_numpy()[0:12,1:13]
# mm_AP   = data[0:12,0]
# a,b     = data.shape 
# print (mm_AP,data,a,b)
# # print (data,a,b)

# carrierFile  = pd.read_csv('t3_carrierc.csv', sep=',')
# carrierdata    = carrierFile.to_numpy()[0:12,1:13]
# mm_AP   = carrierdata[0:12,0]
# a,b     = carrierdata.shape 

# fonts = 12
# 
# img = (data * 255 / np.max(data)).astype('uint8')
# fig = plt.figure(figsize=(6,5))
# ax  = fig.add_subplot(111)
# im  = ax.imshow(img,cmap='inferno',extent=[-0.5,5,2.5,-3],alpha=1.0 )
# cbar = fig.colorbar(im,fraction=0.046, pad=0.04)
# cbar.set_label('$\mu V$', rotation=90,fontsize=fonts)
# ax.set_title('$\Delta f$ ',fontsize=fonts)
# ax.set_xlabel('mm',fontsize=fonts)
# ax.set_ylabel('mm',fontsize=fonts)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plot_filename = 'saline_calibration_map_df'
# plt.savefig(plot_filename +".png",transparent=True,
#            pad_inches=0,bbox_inches='tight')
# plt.show()

# # 
# carrier_img = (carrierdata * 255 / np.max(carrierdata)).astype('uint8')
# fig = plt.figure(figsize=(6,5))
# ax  = fig.add_subplot(111)
# im = ax.imshow(carrier_img,cmap='inferno',extent=[-0.5,5,2.5,-3],alpha=1.0)
# cbar = fig.colorbar(im,fraction=0.046, pad=0.04)
# ax.set_title('$f@ 500kHz$ ',fontsize=fonts)
# cbar.set_label('$\mu V$', rotation=90,fontsize=fonts)
# ax.set_xlabel('mm',fontsize=fonts)
# ax.set_ylabel('mm',fontsize=fonts)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plot_filename = 'saline_calibration_map_carrier'
# plt.savefig(plot_filename +".png",transparent=True,
#            pad_inches=0,bbox_inches='tight')
# plt.savefig(plot_filename)
# plt.show()

