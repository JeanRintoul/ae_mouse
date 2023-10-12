import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt


# for the amplitude, I take the lfp amplitude from 0.5-300Hz filter. 
frequencies 	   = [1,2,5,10,40,100]
# phantom_amplitudes = [22.9,8.94,15,11.3,13.46]  # 

# frequencies 	   = [1,2(150),5(),10(150),30(561),100(173)] # e107
# mouse_amplitudes   = [154,95,46,40.3,41]        # e107

# f: 	5   /  10    /  30   / 100   e 107 t5 
# a: 	790 /  600   / 560   / 584  in fft 

# phantom, e107, t6
frequencies2 		= [2,5,10,40,100]
# phantom_amplitudes = [750,1193,1058,861,780]
phantom_amplitudes = [2950,1900,1280,1680,1900]
mouse_amplitudes   = [3504,2806,3254,3221,3387,2824 ]        # e113
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(frequencies2, phantom_amplitudes,'k')
plt.plot(frequencies,mouse_amplitudes,'r')
plt.plot(frequencies2, phantom_amplitudes,'.k')
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
