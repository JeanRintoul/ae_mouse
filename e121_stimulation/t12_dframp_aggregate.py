'''

Title: compare the data going into generator with the data coming out of generator. 

Author: Jean Rintoul
Date:   26.10.2023

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy.signal import hilbert
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
# There appears to be no phase offset. 
# I want to save out the:  dlfp_data, and the 3 peak to peak measures, dfx, filenumber. 
# I also want to group it into an array, so I think I need a nested loop. 
# There would be 2 output npzs. One for mouse, one for phantom. 
# 
df_ramp_frequencies  = [1,2,5,10,40,100,300,1000]
# savepath             = 'D:\\ae_mouse\\e121_stimulation\\t1_mouse\\'
# # mouse df ramp files. 
# df_ramp_files1       = [3,4,5,6,7,8,9,10]
# df_ramp_files2       = [18,19,20,21,22,23,24,25]
# df_ramp_files3       = [26,27,28,29,30,31,32,33]
# df_ramp_files4       = [34,35,36,37,38,39,40,41]


# phantom df ramp files. 
savepath       = 'D:\\ae_mouse\\e121_stimulation\\t2_phantom\\'
df_ramp_files1       = [1,2,3,4,5,6,7,8]
df_ramp_files2       = [9,10,11,12,13,14,15,16]
df_ramp_files3       = [17,18,19,20,21,22,23,24]
df_ramp_files4       = [25,26,11,28,29,30,31,32]

df_ramp_file_array = [df_ramp_files1,df_ramp_files2,df_ramp_files3,df_ramp_files4]
df_ramp_file_array = np.array(df_ramp_file_array)
a,b = df_ramp_file_array.shape

final_data = []
data_array = []
print ('a,b',a,b)
pp1 = np.zeros((a,b))
pp2 = np.zeros((a,b))
pp3 = np.zeros((a,b))
# 60000 is the length of the lfp data after decimation
lfp_aggregates = np.zeros((a,b,60000))
print ('pp1 shape,a,b', pp1.shape,a,b )
# 
duration        = 6.0   
m_channel       = 0 
rf_channel      = 2 
gain            = 500 
Fs              = 1e7
timestep        = 1.0/Fs
N               = int(Fs*duration)
t               = np.linspace(0, duration, N, endpoint=False)
# 
# 
for i in range(a):
    for j in range(b):  
 
        print ('i,j: ',i,j)
        element = j
        file_number = df_ramp_file_array[i,j]
        #  
        dfx         = df_ramp_frequencies[element] 
        print ('dfx/file: ',dfx,file_number)
        filename    = savepath + 't'+str(file_number)+'_stream.npy'
        data        = np.load(filename)
        fsignal     = 1e6*data[m_channel]/gain
        rfsignal    = 10*data[rf_channel]  
        fft_start   = int(2*Fs)
        fft_end     = int(4*Fs) 
        fft_data        = fft(fsignal[fft_start:fft_end])
        N = fft_end - fft_start
        fft_data        = np.abs(2.0/(N-1) * (fft_data))[1:(N-1)//2]
        xf              = np.fft.fftfreq( (N-1), d=timestep)[:(N-1)//2]
        frequencies     = xf[1:(N-1)//2]
        # dfx band filter. An alternate peak-peak measure. 
        bandwidth = 10
        bl = dfx - bandwidth
        if (dfx-bandwidth) <= 0:
            bl = 0.1
        bh = dfx + bandwidth
        sos_dfx_band = iirfilter(17, [bl,bh], rs=60, btype='bandpass',
                               analog=False, ftype='cheby2', fs=Fs,
                               output='sos')
        # lfp filter. Since some of my measures are about 1000Hz, It may be better not to use this filter. 
        df_h = 1100
        sos_lfp_band = iirfilter(17, [df_h], rs=60, btype='lowpass',
                               analog=False, ftype='cheby2', fs=Fs,
                               output='sos')

        mains_cut_band = iirfilter(17, [48,52], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
        fsignal              = sosfiltfilt(mains_cut_band, fsignal)
        lfp_data              = sosfiltfilt(sos_lfp_band, fsignal)
        # spike_data            = sosfiltfilt(sos_spike_band, fsignal)
        dfx_data              = sosfiltfilt(sos_dfx_band, fsignal)
        # decimate the data so it plots faster. 
        new_Fs                        = 1e4
        downsampling_factor           = int(Fs/new_Fs)
        dt           = t[::downsampling_factor]
        dlfp_data    = lfp_data[::downsampling_factor]
        # dspike_data  = spike_data[::downsampling_factor]
        ddfx_data = dfx_data[::downsampling_factor]
        # 
        df_idx = find_nearest(frequencies,dfx)
        lfp_height = np.max(fsignal[fft_start:fft_end]) - np.min(fsignal[fft_start:fft_end])
        dfx_filtered_height = np.max(ddfx_data) - np.min(ddfx_data)
        print ('fft df p-p, measured p-p',np.round(2*fft_data[df_idx],2),np.round(lfp_height,2 ),np.round(dfx_filtered_height,2) )
        # pp1, lfp_aggregates. 
        pp1[i,j] = np.round(2*fft_data[df_idx],2)
        pp2[i,j] = np.round(lfp_height,2 )
        pp3[i,j] = np.round(dfx_filtered_height,2)
        lfp_aggregates[i,j,:] = dlfp_data
# 
# 
outfile                      = 'phantom_data.npz'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,pp1=pp1,pp2=pp2,pp3=pp3,df_ramp_file_array=df_ramp_file_array,df_ramp_frequencies=df_ramp_frequencies,lfp_aggregates=lfp_aggregates)
print ('saved out a data file!')
# 
# 
# start = 0 
# stop  = duration
# # 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(dt,dlfp_data,'k')
# ax.set_xlim([start,stop])
# ax2 = fig.add_subplot(312)
# plt.plot(frequencies,fft_data,'k')
# ax2.set_xlim([0,dfx+40])
# # plt.legend(['from generator'],loc='upper right')
# ax3 = fig.add_subplot(313)
# plt.plot(dt,ddfx_data,'k')
# # ax3.set_xlim([start,stop])
# # plt.legend(['from recording chan'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plot_filename = 'whole_file_comparison.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()
# # 

# analytical_signal  = hilbert(fsignal )
# amplitude_envelope = np.abs(analytical_signal)

# rfanalytical_signal  = hilbert(rfsignal )
# rfamplitude_envelope = np.abs(rfanalytical_signal)
# sos_lp = iirfilter(17, [2000], rs=60, btype='lowpass',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')

# hilberted               = sosfiltfilt(sos_lp, amplitude_envelope)
# rfhilberted             = sosfiltfilt(sos_lp, rfamplitude_envelope)

# # How do we analyze the phase offsets? 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot(t,rfsignal,'r')
# # plt.plot(t,np.real(analytical_signal),'r')
# ax.set_xlim([start,stop])
# ax2 = fig.add_subplot(312)
# plt.plot(t,lfp_data,'k')
# ax2.set_xlim([start,stop])
# ax3 = fig.add_subplot(313)
# plt.plot(t,hilberted/np.max(hilberted),'k')
# plt.plot(t,rfhilberted/np.max(rfhilberted),'r')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plt.show()



# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(211)
# plt.plot(t,from_generator,'k')
# ax.set_xlim([start,stop])
# plt.legend(['from generator'],loc='upper right')
# ax2 = fig.add_subplot(212)
# plt.plot(t,from_recording_chan,'k')
# ax2.set_xlim([start,stop])
# plt.legend(['from recording chan'],loc='upper right')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plot_filename = 'zoom_file_comparison.png'
# plt.savefig(plot_filename, bbox_inches="tight")
# plt.show()

