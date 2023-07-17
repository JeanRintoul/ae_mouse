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
import os
import time 

fonts = 14
matplotlib.rcParams.update({'font.size': fonts})

# 
start_files     = 2
end_files       = 25
duration        = 4 
Fs              = 5e6 
N               = int(Fs*duration)
start_pause     = int(0.25 * N+1)
end_pause       = int(0.875 * N-1)
window          = 1 
v_channel       = 6 
carrier_f       = 500000
v_freq          = 8000
timestep        = 1.0/Fs

def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

files       = np.linspace(start_files,end_files,(end_files-start_files+1),dtype=np.int16)
savepath    = 'D:\\mouse_aeti\\e97_MEPS\\t5\\'

xf              = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies     = xf[1:(end_pause-start_pause)//2]
US_idx  = find_nearest(frequencies,int(carrier_f) )
df_idx  = find_nearest(frequencies,carrier_f-v_freq)
sf_idx  = find_nearest(frequencies,carrier_f+v_freq)

time_zero   = 0 
things      = []
for i in range(len(files) ):
    print ('file number:', files[i]) 
    filename     = savepath + 't'+str(files[i])+'_stream.npy'
    time_passed  = os.path.getctime(filename)
    if i == 0:
        time_zero = os.path.getctime(filename)
    time_created = int(time_passed - time_zero)
    print ('time created:',time_created)
    data = np.load(filename)
    a,b = data.shape
    t = np.linspace(0, duration, N, endpoint=False)
    # convert it to microvolts by taking the gain into account. 
    mdata    = 1e6*data[v_channel]
    fft_data = fft(mdata[start_pause:end_pause]*window)
    fft_data = np.abs(2.0/(end_pause-start_pause) * (fft_data))[1:(end_pause-start_pause)//2]
    # 

    things.append([fft_data[US_idx], fft_data[df_idx],fft_data[sf_idx],time_created]) 

# 
things = np.array(things).T
no_files,thing_items = things.shape
print ('things shape',things.shape)

US     = things[0,:]
df     = things[1,:]
sf     = things[2,:]
times  = things[3,:]


# 
fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111)
ax.plot(times,df,'k')
ax.plot(times,US,'r')
ax.plot(times,sf,'gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Times (s)')
ax.set_ylabel('$\mu V$')
plt.legend(['$\Delta f$','$f@ 500kHz$','$\Sigma f$'],loc='upper left',framealpha=0.0)
plot_filename = 'time_wrt_ae_saline'
plt.savefig(plot_filename +".png",transparent=True,
           pad_inches=0,bbox_inches='tight')
plt.show()

