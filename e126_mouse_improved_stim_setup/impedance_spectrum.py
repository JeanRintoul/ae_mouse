'''

Title: vep signal inspection
Function: takes a single file, and averages the veps to see them better. 

Author: Jean Rintoul
Date: 23.10.2022

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.stats import ttest_ind
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16
#
# File list:  t1
frequency = np.array([10,1000,10000,100000,500000,1000000,1500000,2000000])/1e6

impedance = np.array([[8014.9,	7881.9],
[3940,	4143],
[3196.99,	3181],
[1674.46,	1670.16],
[604.57,	606.93],
[363.68,	376.06],
[310.3,	257.44],
[223.82,	205.56]])
#


mean_impedance = np.mean(impedance,1)
std_dfs = np.std(impedance,1)


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(frequency,mean_impedance,'k',marker='.')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
# ax.set_xlim([0-0.0,np.max(salinity)+0.1])
plot_filename ='mouse_impedance.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()



