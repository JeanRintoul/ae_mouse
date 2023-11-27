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

mouse_impedance = np.array([[8014.9,	7881.9],
[3940,	4143],
[3196.99,	3181],
[1674.46,	1670.16],
[604.57,	606.93],
[363.68,	376.06],
[310.3,	257.44],
[223.82,	205.56]])
#
saline_impedance = np.array([[1151.07,	1158.1089],
[486.32	,482.9],
[381.47,	381.39],
[361.2,	360.85],
[344.64,	348.36],
[325.3,	324.29],
[290.44	,325.24],
[240.07,	275.5]  ])



mean_mouse_impedance = np.mean(mouse_impedance,1)
mean_saline_impedance = np.mean(saline_impedance,1)


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.plot(frequency,mean_mouse_impedance,'k',marker='.')
plt.plot(frequency,mean_saline_impedance,'r',marker='.')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
ax.set_ylim([0,8500])
plt.legend(['mouse','saline'],loc='upper right')
# ax.set_xlim([0-0.0,np.max(salinity)+0.1])
plot_filename ='comparative_impedance.png'
plt.savefig(plot_filename, bbox_inches="tight")
plt.show()



