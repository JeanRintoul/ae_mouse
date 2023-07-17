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

# dv_position = [4,3.5,3,2.5,2,1.5,1,0.5,0] 
# df_dv       = [6,11,13,51,25,22,21,20,18]
# sf_dv       = [6,9,12,49,25,22,22,20,20]
# carrier_dv  = [451,483,537,477,550,523,605,572,593]


# fig = plt.figure(figsize=(8,4))
# ax  = fig.add_subplot(111)
# ax.plot(dv_position,df_dv,'k')
# # ax.plot(dv_position,carrier_dv,'r')
# ax.plot(dv_position,sf_dv,'gray')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xlabel('Vertical Distance DV traversal (mm)')
# ax.set_ylabel('$\mu V$')
# plt.legend(['$\Delta f$','$\Sigma f$'],loc='upper left',framealpha=0.0)
# plot_filename = 'dvposition_vs_ae'
# plt.savefig(plot_filename +".png",transparent=True,
#            pad_inches=0,bbox_inches='tight')
# plt.show()

# changing the voltage
v_out       = [0,0.5,1,2,4,8,12]
# with silicon
df_s          = [0.4,6,13,21,42.7,90,123.9]
carrier_s     = [41,36,43,41,44.9,137,131]
sf_s          = [1.1,4.6,10.8,22.5,45.6,74,130.6]
# without silicon
df          = [1.8,5.4,12,25.9,41.45,95.4,141.7]
carrier     = [52,41,70,39.8,40.4,38.8,57.7]
sf          = [0.34,6.3,10.8,22.5,46.0,94.4,142.7]

# changing the pressure
p_out       = [0,0.01,0.05,0.08,0.1,0.11,0.15,0.2]
pressure    = [0,0.1,0.6,0.9,1.0,1.25,1.6,2.1]  # MPa
# with silicon
p_df_s        = [0.9,1.45,7.9,9.2,12.8,12.9,15.8,16.6]
p_carrier_s   = [0.9,5.22,21.5,65,42.3,53,58,80]
p_sf_s        = [1,1.55,4.67,8.52,11.2,10.4,15.06,17.5]
# without silicon
p_df        = [0.34,1.22,6.98,9.7,11.2,11.9,15.48,18.6]
p_carrier   = [0.5,3.5,16.65,40,100,40.48,96.9,151.2]
p_sf        = [0.3,1.31,6.05,8.9,10.6,12.18,15.77,18.59]

# 
# 
fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111)
ax.plot(v_out,df,'k')
ax.plot(v_out,df_s,'r')
# ax.plot(v_out,carrier,'r')
# ax.plot(v_out,sf,'gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Voltage output(V)')
ax.set_ylabel('$\mu V$')
plt.legend(['$\Delta f$ no silicon','$\Delta f$ with silicon'],loc='upper left',framealpha=0.0)
# plt.legend(['$\Delta f$','$f@ 500kHz$','$\Sigma f$'],loc='center left',framealpha=0.0)
plot_filename = 'gradient_wrt_voltage'
plt.savefig(plot_filename +".png",transparent=True,
           pad_inches=0,bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111)
# ax.plot(pressure,p_df,'g')
ax.plot(pressure,p_df,'k')
ax.plot(pressure,p_df_s,'r')
# ax.plot(pressure,p_carrier,'r')
# ax.plot(pressure,p_sf,'gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Pressure Output (MPa)')
ax.set_ylabel('$\mu V$')
plt.legend(['$\Delta f$ no silicon','$\Delta f$ with silicon'],loc='upper left',framealpha=0.0)
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

