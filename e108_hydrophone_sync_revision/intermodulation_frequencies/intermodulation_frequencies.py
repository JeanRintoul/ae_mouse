import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq,fftshift,ifft,ifftshift
from scipy.signal import blackman
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.signal import fftconvolve
#
# Design a band pass filter to isolate just the information I want to get the peak to peak amplitude more accurately.
# 
# 
# // Channel identities: 
# // 1. rf amplifier output
# // 2. hydrophone probe
# // 

# // 3. current monitor for e1
# // 4. current monitor for e2

# // 5. v mon e1 x10  
# // 6. v mon e2 x10 

# // 7. tiepie voltage output waveform x10 
# // 8. diff input voltage across 1k resister. x 10
Fs = 5e6
duration = 3.0
N = Fs*duration
print("expected no. samples:",N)
t = np.linspace(0, duration, int(N), endpoint=False)
resistor_current_mon = 49.9  #  49.9 Ohms for current monitor, 1k resistor 
timestep = 1.0/Fs
N = int(N)
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

rf_channel  = 0    
hydrophone_channel = 1 
i_channel   = 3
v_channel   = 4 
ae_channel  = 6  
e_channel   = 7 
gain        = 100


filename = 'old_ati_stream_data.npy'

d = np.load(filename)
a,b,c = d.shape
data = d.transpose(1,0,2).reshape(b,-1) 
# print (data.shape)

fft_aefield = fft(1e6*data[ae_channel]/gain)
fft_ae = np.abs(2.0/N * (fft_aefield))[1:N//2]

fft_efield = fft(10*data[e_channel])
fft_e = np.abs(2.0/N * (fft_efield))[1:N//2]

fft_rf = fft(data[rf_channel])
fft_rf = np.abs(2.0/N * (fft_rf))[1:N//2]

fft_ifield = fft(5*data[i_channel]/resistor_current_mon)
fft_ifield = np.abs(2.0/N * (fft_ifield))[1:N//2]


fig = plt.figure(figsize=(7.5,5))
ax = fig.add_subplot(111)
plt.plot(frequencies, fft_ae, color = 'k')
# plt.xlim([450000,550000])
plt.xlim([1e6+0,1e6+50000])
# plt.xlim([0,50000])
plt.ylim([0,50])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("fft_mixed_signal.png", bbox_inches="tight") 
plt.show()




