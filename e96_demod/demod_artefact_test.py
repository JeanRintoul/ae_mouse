"""
Read in the summary files, 

"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('D:\\mouse_aeti')  #  so that we can import from the parent folder. 
import mouse_library as m

#  this data is somewhat filtered. 
filename          	= 'demod_pressure_summary_data.npz'
# filename          	= 'demod_pressure_summary_data300f3.npz'
data              	= np.load(filename)
d_total_rd       	= np.array(data['total_rd'])
d_total_dd          = np.array(data['total_dd'])
d_frequencies       = data['frequencies']
d_total_rfft        = np.array(data['total_rfft'])
d_total_dfft        = np.array(data['total_dfft'])
marker       		= data['total_mm']

a,b = d_total_rd.shape
print ('with pressure shape',d_total_rd.shape)

nfilename          	= 'demod_nopressure_summary_data.npz'
# nfilename          	= 'demod_nopressure_summary_data300f3.npz'
ndata              	= np.load(nfilename)
n_total_rd       	= np.array(ndata['total_rd'])
n_total_dd          = np.array(ndata['total_dd'])
n_frequencies       = ndata['frequencies']
n_total_rfft        = np.array(ndata['total_rfft'])
n_total_dfft        = np.array(ndata['total_dfft'])
nmarker       		= ndata['total_mm']

c,d = n_total_rd.shape
print ('without pressure shape',n_total_rd.shape)
# 
n_events,N 	= d_total_rd.shape
Fs 			= 1e7 
timestep 	= 1/Fs 
duration 	= timestep*N
t        	= np.linspace(0, duration, N, endpoint=False)
# 
# pressure applied with acoustic connection
average_VEP 		= np.mean(d_total_rd,axis=0)
std_VEP     		= np.std(d_total_rd,axis=0)
average_demod_VEP 	= np.mean(d_total_dd,axis=0)
std_demod_VEP     	= np.std(d_total_dd,axis=0)
average_VEP_fft 	= np.mean(d_total_rfft,axis=0)
average_demod_fft 	= np.mean(d_total_dfft,axis=0)
std_VEP_fft 		= np.std(d_total_rfft,axis=0)
std_demod_fft 		= np.std(d_total_dfft,axis=0)

# no acoustic connection 
n_average_VEP 			= np.mean(n_total_rd,axis=0)
n_std_VEP     			= np.std(n_total_rd,axis=0)
n_average_demod_VEP 	= np.mean(n_total_dd,axis=0)
n_std_demod_VEP     	= np.std(n_total_dd,axis=0)
n_average_VEP_fft 		= np.mean(n_total_rfft,axis=0)
n_average_demod_fft 	= np.mean(n_total_dfft,axis=0)
n_std_VEP_fft 			= np.std(n_total_rfft,axis=0)
n_std_demod_fft 		= np.std(n_total_dfft,axis=0)
# 
fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
ax.plot(t,np.max(average_demod_VEP)*marker/np.max(marker),'g')
ax.plot(t,average_demod_VEP/np.max(average_demod_VEP),'k')
ax.plot(t,n_average_demod_VEP/np.max(n_average_demod_VEP),'r')
ax.plot(t,average_VEP/np.max(average_VEP),'b')
# 
# plt.fill_between(t, n_average_demod_VEP - n_std_demod_VEP, n_average_demod_VEP + n_std_demod_VEP,
#                   color='r', alpha=0.5)
# plt.fill_between(t, average_demod_VEP - std_demod_VEP, average_demod_VEP + std_demod_VEP,
#                   color='k', alpha=0.5)
# 
# plt.legend(['led on/off','acoustic connection','no acoustic_connection','real VEP'])
plt.legend(['led on/off','acoustic connection','no acoustic_connection','vep'])
ax.set_xlim([0,np.max(t)])
ax.set_ylabel('norm Volts ($\mu$V)')
ax.set_xlabel('Time(s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = 'demod_vep_individual_trials.png'
plt.savefig(plot_filename)
plt.show()


fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111)
ax.plot(t,np.max(average_demod_VEP)*marker,'g')
ax.plot(t,average_demod_VEP,'k')
ax.plot(t,n_average_demod_VEP,'r')
ax.plot(t,average_VEP,'b')

# plt.fill_between(t, n_average_demod_VEP - n_std_demod_VEP, n_average_demod_VEP + n_std_demod_VEP,
#                   color='r', alpha=0.5)
# plt.fill_between(t, average_demod_VEP - std_demod_VEP, average_demod_VEP + std_demod_VEP,
#                   color='k', alpha=0.5)


# plt.legend(['led on/off','acoustic connection','no acoustic_connection','real VEP'])
plt.legend(['led on/off','acoustic connection','no acoustic_connection','vep'])
ax.set_xlim([0,np.max(t)])
ax.set_ylabel('norm Volts ($\mu$V)')
ax.set_xlabel('Time(s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_filename = 'demod_vep_individual_trials_unscaled.png'
plt.savefig(plot_filename)
plt.show()

# # FFT 
fig = plt.figure(figsize=(10,6))
ax2  = fig.add_subplot(111)
plt.plot(d_frequencies,average_demod_fft/a,'k')
plt.plot(n_frequencies,n_average_demod_fft/c,'r')
plt.legend(['acoustic connection','no acoustic_connection'])
ax2.set_xlim([0,np.max(d_frequencies)])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plot_filename = 'vep_individual_trials_fft.png'
plt.savefig(plot_filename)
plt.show()
