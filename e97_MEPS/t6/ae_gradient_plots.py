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

dv_position = [4,3.5,3,2.5,2,1.5,1,0.5,0] 
df_dv       = [6,11,13,51,25,22,21,20,18]
sf_dv       = [6,9,12,49,25,22,22,20,20]
carrier_dv  = [451,483,537,477,550,523,605,572,593]


fig = plt.figure(figsize=(8,4))
ax  = fig.add_subplot(111)
ax.plot(dv_position,df_dv,'k')
# ax.plot(dv_position,carrier_dv,'r')
ax.plot(dv_position,sf_dv,'gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Vertical Distance DV traversal (mm)')
ax.set_ylabel('$\mu V$')
plt.legend(['$\Delta f$','$\Sigma f$'],loc='upper left',framealpha=0.0)
plot_filename = 'dvposition_vs_ae'
plt.savefig(plot_filename +".png",transparent=True,
           pad_inches=0,bbox_inches='tight')
plt.show()


v_out       = [0,0.5,1,2,4,8,12]
df          = [0.4,14.9,34,65.6,122.6,238.6,335.6]
carrier     = [639,532,596,601,691,635,675]
sf          = [1.9,14.1,32,65.19,117.8,239.8,349]
# 
#
p_out       = [0,0.01,0.05,0.08,0.1,0.11,0.15]
p_df        = [1.14,1.8,6,21.8,30,30.2,37.9]
p_carrier   = [1.2,50.9,148.7,464.6,639,620,833]
p_sf        = [0.47,4.5,4.7,23.4,29,30.9,35.02]
pressure    = [0,0.1,0.6,0.9,1.0,1.25,1.6]  # MPa
# 
# 
fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111)
ax.plot(v_out,df,'k')
# ax.plot(v_out,carrier,'r')
ax.plot(v_out,sf,'gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Voltage output(V)')
ax.set_ylabel('$\mu V$')
plt.legend(['$\Delta f$','$\Sigma f$'],loc='upper left',framealpha=0.0)
# plt.legend(['$\Delta f$','$f@ 500kHz$','$\Sigma f$'],loc='center left',framealpha=0.0)
plot_filename = 'gradient_wrt_voltage'
plt.savefig(plot_filename +".png",transparent=True,
           pad_inches=0,bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111)
# ax.plot(pressure,p_df,'g')
ax.plot(pressure,p_df,'k')
# ax.plot(pressure,p_carrier,'r')
ax.plot(pressure,p_sf,'gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Pressure Output (MPa)')
ax.set_ylabel('$\mu V$')
plt.legend(['$\Delta f$','$\Sigma f$'],loc='upper left',framealpha=0.0)
# plt.legend(['$\Delta f$','$f@ 500kHz$','$\Sigma f$'],loc='upper left',framealpha=0.0)
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

