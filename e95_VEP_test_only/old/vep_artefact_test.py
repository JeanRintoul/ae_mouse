'''

Title: Plot the veps versus the blocked LED with mean and standard deviation. Also show the raw LED flashes. 
Author: Jean Rintoul
Date: 21.02.2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
import pandas as pd
import sys
sys.path.append('D:\\mouse_aeti')  #  so that we can import from the parent folder. 
import mouse_library as m
# 
# 
# number of flashes chosen 
number_of_flashes = 50
filename          = 'aggregated_veps.npz'
data              = np.load(filename)
vep_epochs        = data['epochs']
markers           = data['markers']
time              = data['time_segment']

time_offset = 0.0626
time = time - time_offset
print (vep_epochs.shape, markers.shape,len(time))
# Read in the second file. 
nled_filename          = 'aggregated_noled_veps.npz'
nled_data              = np.load(nled_filename)
nled_vep_epochs        = nled_data['epochs']
nled_markers           = nled_data['markers']
nled_time              = nled_data['time_segment']
marker                 = 10*np.mean(markers,axis=0)

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(time,vep_epochs.T)
plt.plot(time,marker,'k')
ax.set_xlim([np.min(time),np.max(time)])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time(s)')
plot_filename = 'vep_individual_trials.png'
plt.savefig(plot_filename)
plt.show()

vep_epochs = vep_epochs[0:number_of_flashes,:]
nled_vep_epochs = nled_vep_epochs[0:number_of_flashes,:]

# # # # # # # # # # # # # # # # # # # # 
average_vep = np.mean(vep_epochs,axis=0)
std_vep     = np.std(vep_epochs,axis=0)

average_noled = np.mean(nled_vep_epochs,axis=0)
std_noled     = np.std(nled_vep_epochs,axis=0)
# # # # # # # # # # # # # # 



fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
plt.plot(time,average_vep,'r')
plt.plot(time,average_noled,'k')
plt.plot(time,marker,'g')
plt.fill_between(time, average_vep - std_vep, average_vep + std_vep,
                  color='red', alpha=0.5)

plt.fill_between(time, average_noled - std_noled, average_noled + std_noled,
                  color='k', alpha=0.5)
plt.plot(time,marker,'g')
ax.set_xlim([np.min(time),np.max(time)])
ax.set_ylabel('Volts ($\mu$V)')
ax.set_xlabel('Time(s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(['average vep (n='+str(number_of_flashes)+')','average no led (n='+str(number_of_flashes)+')','led marker'])
plot_filename = 'vep_artefact_test.png'
plt.savefig(plot_filename)
plt.show()
