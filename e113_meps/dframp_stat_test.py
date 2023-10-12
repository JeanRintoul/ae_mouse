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

# from t2. with F21. 
df_frequencies  	= [1,2,5,10,40,100]
mouse_dfs 			= [1934.2,1425,2153,1689,1676,1542]
phantom_dfs 		= [4356,3576,4628,4352, 2383  ,4983 ]
# 
# 
# 
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
plt.plot(df_frequencies,mouse_dfs,'r')
plt.plot(df_frequencies,phantom_dfs,'k')
plt.legend(['Mouse','Phantom'],loc='upper right')
# plt.xlim([0,5000])
# plt.ylim([0,0.001])
# ax.set_xlabel('frequencies(Hz)')

plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show
plt.tight_layout()
plt.savefig('dframp.png')
plt.show()


