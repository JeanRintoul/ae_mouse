import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
import os
import time 


filename    = 'single_wave.npz'
data = np.load(filename)
d = data['d']

d_f 	= d[:,0]
d_rf 	= d[:,1]
d_mf 	= d[:,2]



fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
plt.plot(d_f,d_rf,'-k.')
ax.set_ylabel('Volts (V)')
plt.legend(['rf monitor'],loc='upper right')
ax.set_xlim([0,1000010])
ax2 = fig.add_subplot(212)
plt.plot(d_f,d_mf,'-r.')
plt.legend(['measured carrier amplitude@electrode'],loc='upper right')
ax2.set_ylabel('Volts ($\mu$V)')
ax2.set_xlabel('Frequency(Hz)')
ax2.set_xlim([0,1000010])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.suptitle('RF versus measured carrier response amplitude')
plot_filename = savepath + '\\t'+str(test_no)+'_rf_response.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(d_f,d_rf/np.max(d_rf),'-k.')
plt.plot(d_f,d_mf/np.max(d_mf),'-r.')
ax.set_xlim([0,1000010])
ax.set_ylabel('Volts (V)')
plt.legend(['rf monitor','measured carrier amplitude@electrode'],loc='upper right')
ax2.set_ylabel('normalized')
ax2.set_xlabel('Frequency(Hz)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.suptitle('RF versus measured carrier response shape')
plot_filename = savepath + '\\t'+str(test_no)+'_rf_response_normalized.png'
plt.savefig(plot_filename)
plt.show()















