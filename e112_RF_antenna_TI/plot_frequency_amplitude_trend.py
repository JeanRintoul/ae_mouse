import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt


# 
frequencies 	   = [3,5,10,40,100]
phantom_amplitudes = [22.9,8.94,15,11.3,13.46] 
mouse_amplitudes   = [154,95,46,40.3,41]


# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(frequencies, phantom_amplitudes,'k')
plt.plot(frequencies,mouse_amplitudes,'r')
plt.plot(frequencies, phantom_amplitudes,'.k')
plt.plot(frequencies,mouse_amplitudes,'.r')

plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.xticks([])
# plt.xticks(fontsize=16)
plt.legend(['phantom','mouse'],loc='upper right',fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('frequency_amplitude_plot.png')
plt.show()
