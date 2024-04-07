import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2


prfs = [1,2,5,10,40,80,220,1020]
# US only. 
prf_us = [742,373,249,101,18,13,14,16]
prf_f21us = [291,156,108,50,9.5,8.8,10,10]
prf_f21use = [271,173,150,92,80,102,101.5,97.5]


prf_use = [840,698,1014,830,813,1015,1013,1016]


fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)

plt.plot(prfs,prf_us,'k')
plt.plot(prfs,prf_f21us,'r')
# plt.plot(prfs,prf_f21use,'m')
# ax.set_xlim([0,bands])
# ax.set_yscale('log')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Frequency of PRF (Hz)',fontsize=16)
ax.set_ylabel('Volts ($\mu$V)',fontsize=16)
plt.title('PRF generated electrical amplitudes',fontsize=16)
plt.tight_layout()
plt.legend(['PRF US','PRF F21 US'],loc='upper right',framealpha = 0.0,fontsize=16)
plot_filename = 'prf_signal.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()