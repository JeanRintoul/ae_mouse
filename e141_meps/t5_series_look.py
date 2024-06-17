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
from scipy.signal import hilbert
from scipy.signal import find_peaks
import scipy.stats
from scipy.stats import pearsonr
import pandas as pd
from scipy import signal
# 
# 
# Try with a differently generated VEP filter. 
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16
# 
saveprefix  = './/images//'
# 
savepath             = 'D:\\ae_mouse\\e141_meps\\t5_ketamine\\mouse_aemep_vs_frequency_wf21\\'
# 
outpath              = 'D:\\ae_mouse\\e141_meps\\'
# 
outname = 'aemeps_ketaminet5'
# 
# 
freqs           = [0.5,1,2,4,8,20,40]
n_repeats       = 1
m_channel       = 0 
gain            = 100 
duration        = 6 
band_limit      = 80 
Fs              = 5e6 
timestep        = 1/Fs
N               = int(duration*Fs)
t               = np.linspace(0, duration, N, endpoint=False)
# 
# 
start_idx       = int(0*Fs) 
end_idx         = int(6*Fs) 
newN            = int(end_idx-start_idx)
xf              = np.fft.fftfreq( (newN), d=timestep)[:(newN)//2]
frequencies     = xf[1:(newN)//2]
# 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   

fs      = [] 
amps    = []
flist   = []
dcs     = []
# 
for i in range(len(freqs)):
    for j in range(n_repeats):
        # print ('i,j',i,j)
        frequency = freqs[i]
        repeat    = j+1
        print ('frequency/rep:',frequency,repeat)
        # Load the file.  
        #filename    = savepath + str(frequency) + 'hz\\' + 't' + str(repeat) + '_stream.npy'
        
        filename    = savepath + str(frequency) + 'hz_ae_mep_g100.npy'
        
        data        = np.load(filename)
        fsignal     = (1e6*data[m_channel]/gain)
        fft_raw     = fft(fsignal[start_idx:end_idx])
        fft_raw     = np.abs(2.0/(newN) * (fft_raw))[1:(newN)//2]

        df_idx      = find_nearest(frequencies,frequency)

        final_idx   = find_nearest(frequencies,50)
        dc_idx      = find_nearest(frequencies,0)
        # 

        # 
        amplitude   = fft_raw[df_idx]
        dc          = fft_raw[dc_idx]
        print ('amplitude:',amplitude*2)
        dcs.append(dc)
        fs.append(frequency)
        amps.append(amplitude)
        flist.append(fft_raw[0:final_idx])
        # 
        # 
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(211)
        plt.plot(t,fsignal,'k')
        ax2 = fig.add_subplot(212)
        plt.plot(frequencies,fft_raw,'k')
        ax2.set_xlim([0,45])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        plt.show()


outfile = outpath + '\\'+ outname +'.npz'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,frequencies = frequencies[0:final_idx],flist=flist,fs=fs,dcs=dcs,amps=amps)
print ('saved out a data file!')



# # 
# filename = savepath + filename
# data        = np.load(filename)
# rawdata_summation     = (1e6*data[m_channel]/gain)

# filenamest = savepath + filenamest
# datast        = np.load(filenamest)
# rawdata_summationst     = (1e6*datast[m_channel]/gain)

# N             = len(rawdata_summation)
# # 
# t               = np.linspace(0, duration, N, endpoint=False)
# # 
# sos_low_band    = iirfilter(17, [band_limit], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')

# sos_emg_band    = iirfilter(17, [20,1000], rs=60, btype='bandpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# # 


# newN = int(end_idx-start_idx)
# print ('New N',newN)

# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx   




# # #
# fft_raw          = fft(rawdata_summation[start_idx:end_idx])
# fft_raw          = np.abs(2.0/(newN) * (fft_raw))[1:(newN)//2]

# fft_rawst        = fft(rawdata_summationst[start_idx:end_idx])
# fft_rawst        = np.abs(2.0/(newN) * (fft_rawst))[1:(newN)//2]


# first_file    = sosfiltfilt(sos_low_band, rawdata_summation)
# second_file    = sosfiltfilt(sos_low_band, rawdata_summationst)

# emg_first_file    = sosfiltfilt(sos_emg_band, rawdata_summation)
# emg_second_file    = sosfiltfilt(sos_emg_band, rawdata_summationst)

# # fontsize control. 
# f = 18

# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(311)
# plt.plot(t,first_file,'k')
# plt.plot(t,second_file,'r')
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)

# ax2  = fig.add_subplot(312)
# plt.plot(frequencies,fft_raw,'k')
# plt.plot(frequencies,fft_rawst,'r')
# ax2.set_xlim([0,band_limit])
# # ax.set_ylim([0,7])

# plt.locator_params(axis='y', nbins=6)
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)

# ax3  = fig.add_subplot(313)
# plt.plot(t,emg_first_file/np.max(emg_first_file),'k')

# plt.plot(t,emg_second_file/np.max(emg_second_file) - 1,'r')

# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)

# # plt.locator_params(axis='x', nbins=5)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)

# plt.tight_layout()
# plot_filename = savepath+'_mep.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(3,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_rawst,'k')
# # plt.plot(frequencies,fft_rawk,'k')
# ax.set_xlim([0,band_limit])
# ax.set_ylim([0,7])
# plt.xticks([10,20,30],fontsize=f)
# plt.yticks(fontsize=f)
# plt.locator_params(axis='y', nbins=6)
# # plt.locator_params(axis='x', nbins=5)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = savepath+'_vep_isolation.png'
# plt.savefig(plot_filename)
# plt.show()



# fig = plt.figure(figsize=(3,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_rawst,'k')
# # plt.plot(frequencies,fft_rawk,'k')
# ax.set_xlim([carrier-band_limit,carrier+band_limit])
# ax.set_ylim([0,300])
# plt.xticks([carrier],fontsize=f)
# plt.yticks(fontsize=f)
# plt.locator_params(axis='y', nbins=6)
# # plt.locator_params(axis='x', nbins=5)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = savepath+'_carrier_amplitude_isolation.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(3,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_raw,'k')
# # plt.plot(frequencies,fft_rawk,'k')
# ax.set_xlim([carrier-band_limit,carrier+band_limit])
# # ax.set_ylim([0,7])
# ax.set_ylim([0,300])
# plt.xticks([carrier],fontsize=f)
# plt.yticks(fontsize=f)
# plt.locator_params(axis='y', nbins=6)
# # plt.locator_params(axis='x', nbins=5)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = savepath+'_carrier_amplitude_10hz.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(8,3))
# ax  = fig.add_subplot(111)
# # plt.plot(frequencies,fft_raw,'r')
# plt.plot(frequencies,fft_rawk,'k')
# ax.set_xlim([carrier-band_limit,carrier+band_limit])
# ax.set_ylim([0,0.02])
# plt.xticks([carrier-10,carrier+10,carrier-20,carrier+20,carrier-30,carrier+30],fontsize=f)
# plt.yticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.ticklabel_format(useOffset=False)
# plt.locator_params(axis='y', nbins=6)
# # plt.locator_params(axis='x', nbins=4)
# plt.tight_layout()
# plot_filename = savepath+'_modulation+10Hz.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(8,3))
# ax  = fig.add_subplot(111)
# # plt.plot(frequencies,fft_raw,'r')
# plt.plot(frequencies,fft_rawkst,'k')
# ax.set_xlim([carrier-band_limit,carrier+band_limit])
# ax.set_ylim([0,0.02])
# plt.xticks([carrier-10,carrier+10,carrier-20,carrier+20,carrier-30,carrier+30],fontsize=f)
# plt.yticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.ticklabel_format(useOffset=False)
# plt.locator_params(axis='y', nbins=6)
# # plt.locator_params(axis='x', nbins=4)
# plt.tight_layout()
# plot_filename = savepath+'_modulation_isolation.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(3,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_rawk,'k')
# plt.plot(frequencies,fft_rawkst,'g')

# ax.set_xlim([carrier-band_limit,carrier+band_limit])
# ax.set_ylim([0,600])
# plt.xticks([carrier],fontsize=f)
# plt.yticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plot_filename = savepath+'_carrier_sidetest_distant.png'
# plt.savefig(plot_filename)
# plt.show()


# fig = plt.figure(figsize=(6,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_rawk,'k')
# plt.plot(frequencies,fft_rawkst,'g')

# ax.set_xlim([carrier-band_limit,carrier+band_limit])
# ax.set_ylim([0,0.08])
# plt.xticks([carrier+10,carrier+20,carrier+30],fontsize=f)
# plt.yticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plot_filename = savepath+'_carrier_sidetest.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(6,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_rawkst,'g')
# plt.plot(frequencies,fft_rawk,'k')
# ax.set_xlim([carrier+8,carrier+41])
# ax.set_ylim([0,0.06])
# plt.xticks([carrier+10,carrier+20],fontsize=f)
# plt.yticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.ticklabel_format(useOffset=False)
# plt.tight_layout()
# plot_filename = savepath+'_carrier_sidetest_zoom.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(3,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_rawk,'k')
# plt.plot(frequencies,fft_rawkst,'g')

# ax.set_xlim([0,band_limit])
# ax.set_ylim([0,3])
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plot_filename = savepath+'_vep_sidetest.png'
# plt.savefig(plot_filename)
# plt.show()

# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(211)
# plt.plot(frequencies,fft_rawkst,'g')
# plt.plot(frequencies,fft_rawk,'k')

# # plt.plot(frequencies,fft_raw,'r')
# ax.set_xlim([carrier-band_limit,carrier+band_limit])
# # plt.axvline(x=carrier-8,color='b')
# # plt.axvline(x=carrier+8,color='b')
# # ax.set_ylim([0,0.05])
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)
# # plt.legend(['Kaiser','Rectangular'],fontsize=f,loc='upper right',framealpha=0.0)
# ax2  = fig.add_subplot(212)
# plt.plot(frequencies,fft_rawkst,'g')
# plt.plot(frequencies,fft_rawk,'k')
# # plt.legend(['Kaiser','Rectangular'],fontsize=f,loc='upper right',framealpha=0.0)
# ax2.set_xlim([0,band_limit])
# # ax2.set_ylim([0,3])
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)
# plt.tight_layout()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = savepath+'_comparison.png'
# plt.savefig(plot_filename)
# plt.show()





# fig = plt.figure(figsize=(3,3))
# ax  = fig.add_subplot(111)
# plt.plot(frequencies,fft_rawk,'k')
# plt.plot(frequencies,fft_raw,'r')
# ax.set_xlim([carrier-20,carrier+20])
# plt.xticks([carrier-20,carrier,carrier+20])
# # ax.set_ylim([0,15])
# ax.ticklabel_format(useOffset=False)
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)
# plt.tight_layout()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plot_filename = savepath+'twotone_carrier.png'
# plt.savefig(plot_filename)
# plt.show()


# 
# 

# 
# t               = np.linspace(0, duration, N, endpoint=False)
# 
#                     
# print ('stuff shape: ',start_indexes)
# # demodulate the signal.
# analytical_signal       = hilbert(carrier_band_summation) # Hilbert demodulate.  
# h_signal                = -np.abs(analytical_signal)
# demodulated_signal      = h_signal - np.mean(h_signal)
# # temp adjust. 
# demodulated_signal  = np.real(analytical_signal)
# # 
# # 
# def demodulate(in_signal,carrier_f,tt): 
#     # return np.abs(in_signal*np.exp(2*np.pi*1j*carrier_f*tt))
#     return np.imag(in_signal*np.exp(2*np.pi*1j*carrier_f*tt))    
# demodulated_signal2       = demodulate(carrier_band_summation,carrier,time_segment) 
# # 
# # calculate the fft of the recovered signal.
# fft_demod         = fft(demodulated_signal)
# fft_demod         = np.abs(2.0/(rawN) * (fft_demod))[1:(rawN)//2]
# # 
# fft_demod2        = fft(demodulated_signal2)
# fft_demod2        = np.abs(2.0/(rawN) * (fft_demod2))[1:(rawN)//2]   
# print ('N lengths',rawN,N)
# # 
# # 
# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(411)
# plt.plot(time_segment,demodulated_signal,'k')
# plt.plot(time_segment,demodulated_signal2,'r')
# ax2 = fig.add_subplot(412)
# plt.plot(time_segment,lfp_summation,'k')
# ax3 = fig.add_subplot(413)
# plt.plot(time_segment,rawdata_summation,'k')
# ax4 = fig.add_subplot(414)
# plt.plot(time_segment,carrier_band_summation,'k')
# plt.show()
# # 
# # NB: It only makes sense to demodulate after much averaging. 
# # Plot the results. 
# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(311)
# plt.plot(raw_frequencies,fft_carrier_band,'k')    
# ax.set_xlim([carrier-h_cut,carrier+h_cut])
# ax.set_ylim([0,20])
# # 
# ax2  = fig.add_subplot(312)
# plt.plot(raw_frequencies,fft_raw,'k')    
# ax2.set_xlim([0,h_cut])
# ax2.set_ylim([0,10])
# # 
# ax3  = fig.add_subplot(313)
# plt.plot(raw_frequencies,fft_demod,'k')   
# plt.plot(raw_frequencies,fft_demod2,'r')   
# ax3.set_xlim([0,20])
# # ax3.set_ylim([0,10])
# plt.show()
# # 
