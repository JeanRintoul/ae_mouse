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
import os
import time 
# 
# 2-25 is acoustic connection
# 26-40 is airgap. 
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

file_list_start = 2
file_list_end   = 25
# file_list_start = 26
# file_list_end   = 40
file_list = np.linspace(file_list_start,file_list_end,(file_list_end-file_list_start+1),dtype=np.int16)
print ('file list:', file_list)

filename    = 'fft_trends_acoustically_connected'
# filename    = 'fft_trends_airgap'
# filename    = 'fft_trends'
filename    = filename + '.npz'
data = np.load(filename)
things = data['things']
no_files,thing_items = things.shape
print ('things shape',things.shape)

times  = things[0,:]
# DC     = things[1,:]
US     = things[2,:]
VEP    = things[3,:]
Delta  = things[4,:]
Thirty = things[5,:]
modulated_Five = things[6,:]
modulated_Thirty = things[7,:]
DC        = things[8,:]

print ('times', times)
print ('DC', DC)
print ('US', US)
print ('VEP', VEP)
print ('Delta', Delta)
print ('Thirty', Thirty)
# 
# these are well correlated. 
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
plt.plot(times,modulated_Five,color='purple')
plt.plot(times,modulated_Thirty,color='magenta')
num = 0 
for i,j in zip(times,modulated_Five):
    ax.annotate('%s' %file_list[num], xy=(i,j)) 
    num = num + 1 
num = 0 
for i,j in zip(times,modulated_Thirty):
    ax.annotate('%s' %file_list[num], xy=(i,j)) 
    num = num + 1 
plt.legend(['modulated 5Hz','modulated 30Hz'])
ax.set_xlabel('Time(s)')
ax.set_ylabel('Volts ($\mu$V)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = filename + '_modulated_trends.png'
plt.savefig(plot_filename)
plt.show()

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
plt.plot(times,US,color='k')
num = 0 
for i,j in zip(times,US):
    ax.annotate('%s' %file_list[num], xy=(i,j))
    num = num + 1 
ax.set_xlabel('Time(s)')
ax.set_ylabel('Volts ($\mu$V)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = filename + '_vs_time.png'
plt.savefig(plot_filename)
plt.show()

# nothing much here. 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(VEP,modulated_Five,'s',color='blue')
# ax2 = fig.add_subplot(212)
# plt.plot(Thirty,modulated_Thirty,'s',color='magenta')
# plt.show()

# well correlated. 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(modulated_Five,modulated_Thirty,'s',color='green')
# plt.show()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(221)
plt.plot(times,US,color='k')
num = 0 
for i,j in zip(times,US):
    ax.annotate('%s' %file_list[num], xy=(i,j))
    num = num + 1 

plt.plot(times,DC,color='r')
plt.plot(times,VEP,color='grey')
plt.plot(times,Delta,color='orange')
plt.plot(times,Thirty,color='cyan')

plt.plot(times,modulated_Five,color='purple')
plt.plot(times,modulated_Thirty,color='magenta')
plt.legend(['US','DC','VEP','Delta','Thirty','m5','m30'],loc = 'upper right',fontsize=6)
ax.set_xlabel('Time(s)')
ax.set_ylabel('Volts ($\mu$V)')

ax2 = fig.add_subplot(222)
plt.plot(Delta,US,'s',color='r')
ax2.set_xlabel('Delta')
ax2.set_ylabel('US')
num = 0 
for i,j in zip(Delta,US):
    ax2.annotate('%s' %file_list[num], xy=(i,j))
    num = num + 1 

ax3 = fig.add_subplot(223)
plt.plot(VEP,Thirty,'s',color='r')
ax3.set_xlabel('VEP @ 5Hz')
ax3.set_ylabel('VEP @ 30Hz')
num = 0 
for i,j in zip(VEP,Thirty):
    ax3.annotate('%s' %file_list[num], xy=(i,j))
    num = num + 1 

ax4 = fig.add_subplot(224)
plt.plot(modulated_Five,modulated_Thirty,'s',color='r')
ax4.set_xlabel('modulated_Five')
ax4.set_ylabel('modulated_Thirty')

# 
num = 0 
for i,j in zip(modulated_Five,modulated_Thirty):
    ax4.annotate('%s' %file_list[num], xy=(i,j))
    num = num + 1 

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)    
# ax.set_xlim([0,1e6])
# ax.set_ylim([0,10.0])
plot_filename = filename + '_plots.png'
plt.savefig(plot_filename)
plt.show()



fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(Delta,modulated_Five,'s',color='r')
ax.set_xlabel('Delta')
ax.set_ylabel('m5')
num = 0 
for i,j in zip(Delta,US):
    ax.annotate('%s' %file_list[num], xy=(i,j))
    num = num + 1 

plot_filename = filename + '_deltavep.png'
plt.savefig(plot_filename)
plt.show()