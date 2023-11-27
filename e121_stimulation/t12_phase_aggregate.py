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
from scipy.signal import find_peaks
from numpy.random import rand
import pandas as pd
import csv
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
# 5905 13701 26000
df_ramp_frequencies = [1,2,5,10,40,100,300,1000]
savepath            = 'D:\\ae_mouse\\e121_stimulation\\t1_mouse\\'
# mouse ramp files. 
df_ramp_files1       = [3,4,5,6,7,8,9,10]
df_ramp_files2       = [18,19,20,21,22,23,24,25]
df_ramp_files3       = [26,27,28,29,30,31,32,33]
df_ramp_files4       = [34,35,36,37,38,39,40,41]


element = 4

# phantom ramp files. 
# savepath       = 'D:\\ae_mouse\\e121_stimulation\\t2_phantom\\'
# df_ramp_files1       = [1,2,3,4,5,6,7,8]
# df_ramp_files2       = [9,10,11,12,13,14,15,16]
# df_ramp_files3       = [17,18,19,20,21,22,23,24]
# df_ramp_files4       = [25,26,11,28,29,30,31,32]
# 
duration        = 6.0   
m_channel       = 0 
rf_channel      = 2 
gain            = 500 
Fs              = 1e7
timestep        = 1.0/Fs
N               = int(Fs*duration)
t = np.linspace(0, duration, N, endpoint=False)
# 
df_ramp_file_array = [df_ramp_files1,df_ramp_files2,df_ramp_files3,df_ramp_files4]
df_ramp_file_array = np.array(df_ramp_file_array)
a,b = df_ramp_file_array.shape


# What am I actually storing up? 
# pp1 = np.zeros((a,b))
peakds = []
peakrs = []
r2s = np.zeros((a,b,29000))
d2s = np.zeros((a,b,29000))

for i in range(a):
    for j in range(b):  
        print ('i,j: ',i,j)
        # temp hack.  
        # i = 0 
        # j = 6 
        element = j
        file_number = df_ramp_file_array[i,j]
        dfx = df_ramp_frequencies[element] 
        print ('dfx/file',dfx,file_number)
        filename    = savepath + 't'+str(file_number)+'_stream.npy'
        data = np.load(filename)

        fsignal             = 1e6*data[m_channel]/gain
        rfsignal            = 10*data[rf_channel]  
        # 
        fft_start = int(1.6*Fs)
        fft_end   = int(4.3*Fs) 
        fft_data        = fft(fsignal[fft_start:fft_end])
        N = fft_end - fft_start
        fft_data        = np.abs(2.0/(N-1) * (fft_data))[1:(N-1)//2]
        xf              = np.fft.fftfreq( (N-1), d=timestep)[:(N-1)//2]
        frequencies     = xf[1:(N-1)//2]
        # dfx band filter. 
        bandwidth = 3
        bl = dfx - bandwidth
        if (dfx-bandwidth) <= 0:
            bl = 0.1
        bh = dfx + bandwidth
        sos_dfx_band = iirfilter(17, [bl,bh], rs=60, btype='bandpass',
                               analog=False, ftype='cheby2', fs=Fs,
                               output='sos')
        # lfp band filter. 
        df_h = 300
        sos_lfp_band = iirfilter(17, [df_h], rs=60, btype='lowpass',
                               analog=False, ftype='cheby2', fs=Fs,
                               output='sos')
        # spike band filter. 
        df_l = 300
        df_h = 1500
        sos_spike_band = iirfilter(17, [df_l, df_h], rs=60, btype='bandpass',
                               analog=False, ftype='cheby2', fs=Fs,
                               output='sos')
        lfp_data              = sosfiltfilt(sos_lfp_band, fsignal)
        spike_data            = sosfiltfilt(sos_spike_band, fsignal)
        dfx_data              = sosfiltfilt(sos_dfx_band, fsignal)

        # decimate the data so it plots faster. 
        new_Fs                        = 1e4
        downsampling_factor           = int(Fs/new_Fs)
        dt           = t[::downsampling_factor]
        dlfp_data    = lfp_data[::downsampling_factor]
        dspike_data  = spike_data[::downsampling_factor]
        ddfx_data = dfx_data[::downsampling_factor]
        # 
        df_idx = find_nearest(frequencies,dfx)
        lfp_height = np.max(lfp_data[fft_start:fft_end]) - np.min(lfp_data[fft_start:fft_end])
        dfx_filtered_height = np.max(ddfx_data) - np.min(ddfx_data)
        print ('fft df p-p, measured p-p',np.round(2*fft_data[df_idx],2),np.round(lfp_height,2 ),np.round(dfx_filtered_height,2) )

        start = 0 
        stop  = duration
        # Hilbert transform the results. 
        rfanalytical_signal  = hilbert(rfsignal )
        rfamplitude_envelope = np.abs(rfanalytical_signal)
        sos_lp = iirfilter(17, [2000], rs=60, btype='lowpass',
                               analog=False, ftype='cheby2', fs=Fs,
                               output='sos')
        rfhilberted             = sosfiltfilt(sos_lp, rfamplitude_envelope)
        rfh_data = rfhilberted [::downsampling_factor]

        # Find the average peak position for this frequency with respect to the original data. 
        start_time = 1.6
        end_time   = 4.5
        start_idx  = find_nearest(dt,start_time)
        end_idx    = find_nearest(dt,end_time)
        # 
        r2 = rfh_data[start_idx:end_idx]
        t2 = dt[start_idx:end_idx]
        d2 = ddfx_data[start_idx:end_idx]
        peaksr, _ = find_peaks(r2, prominence=5)
        peaksd, _ = find_peaks(d2, prominence=5)
        print ('lengths peaksr/peaksd:', len(peaksr),len(peaksd) )
        print ('peaksr: ',peaksr)
        print ('peaksd: ',peaksd)
        # only futz with it if I have to. 
        if (len(peaksr) != len(peaksd) ):
            # 
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(111)
            plt.plot(peaksr,r2[peaksr],'or')
            plt.plot(r2,'r')
            plt.plot(peaksd,d2[peaksd],'ob')
            plt.plot(d2,'b')
            plt.show()

            # number of elements
            n = int(input("Enter number of elements peaksr: "))
            if n > 0:
                if n == 9:
                    # save out peaksr. 
                    df = pd.DataFrame(peaksr)
                    df.to_csv('data.csv', sep=',')               
                    n = int(input("ready?: "))
                    if n == 1: 
                        df = pd.read_csv('data.csv').copy()
                        result = np.array(df.values.tolist()).T
                        result =  result[1,:]
                        peaksr = []
                        for m in range(0, len(result)):
                            if result[m] !=' ':
                                peaksr.append(int(result[m]))
                        print ('newpeaksr: ',peaksr)
                else:
                    # Below line read inputs from user using map() function
                    peaksr = list(map(int, input("\nEnter correct peaksr : ").strip().split()))[:n]
                    peaksr = np.array([int(p) for p in peaksr]) 
                    print("\npeaksr is - ", peaksr)
            # 
            # number of elements
            n = int(input("Enter number of elements peaksd: "))
            if n > 0:
                if n == 9:
                    # save out peaksd. 
                    df = pd.DataFrame(peaksd)
                    df.to_csv('data.csv', sep=',')               
                    n = int(input("ready?: "))
                    if n == 1: 
                        df = pd.read_csv('data.csv').copy()
                        result = np.array(df.values.tolist()).T
                        result =  result[1,:]
                        peaksd = []
                        for m in range(0, len(result)):
                            if result[m] !=' ':
                                peaksd.append(int(result[m]))
                        print ('newpeaksd: ',peaksd)
                else:
                    # Below line read inputs from user using map() function
                    peaksd = list(map(int, input("\nEnter correct peaksd : ").strip().split()))[:n]
                    print("\npeaksd is - ", peaksd)
        # 
        # positive means that the red peak is further to the right of the blue peak
        # Mouse results. 
        # dfx = 1, degree offset = + 30.22
        # dfx = 2, degree offset = + 12.45
        # dfx = 5, degree offset = + 5.73
        # dfx = 10, degree offset = -8.14
        # dfx = 40, degree offset = -2.19
        # 
        # saline
        # dfx = 1  degree offset = +6.8 degrees. 
        # dfx = 2  degree offset = +0.475 degrees. 
        # dfx = 5  degree offset = +1.75 degrees. 
        # dfx = 10  degree offset = +2.31 degrees. 
        # dfx = 40, degree offset = +3.87 degrees. 
        # 
        # peak_offsets        = t2[peaksr]-t2[peaksd]
        # timestep            = t2[2]-t2[1]
        # peak_times          = peak_offsets
        # print ('peak 0times',peak_times)
        # period              = 1/dfx
        # peaks_degree_offset = 360*peak_times/period
        # mean_offset         = np.mean(peaks_degree_offset)
        # print ('peak_times degree offset: ', peaks_degree_offset,mean_offset)
        # print ('dfx: ', dfx)
        # print ('mean degree offset: ', mean_offset)
        # 
        # print ('stuff: ',len(r2),len(d2))
        r2s[i,j] = r2
        d2s[i,j] = d2 
        # apparently a list of lists is the way to go here.
        peakrs.append([i,j,peaksr])
        peakds.append([i,j,peaksd])
   
# I think I should store out r2, d2, peaksr, peaksd.
# I should also on the final run store out t2, frequencies. 
outfile                      = 'mouse_phase_data.npz'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,r2s=r2s,d2s=d2s,peakrs=peakrs,peakds=peakds,t2=t2 )
print ('saved out a data file!')



# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(t2[peaksr], r2[peaksr],'or')
# plt.plot(t2, r2,'r')
# plt.plot(t2[peaksd], d2[peaksd],'ob')
# plt.plot(t2, d2,'b')
# ax.set_xlim([start_time,end_time])
# plt.show()

# 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(t2[peaksr], r2[peaksr],'or')
# plt.plot(t2, r2,'r')
# # 
# plt.plot(t2[peaksd], d2[peaksd],'ob')
# plt.plot(t2, d2,'b')
# # 
# # ax.set_xlim([2,4])
# plt.show()


# # How do we analyze the phase offsets? 
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# plt.plot(dt,ddfx_data/np.max(ddfx_data),'k')
# plt.plot(dt,rfh_data/np.max(rfh_data),'r')

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # ax2.spines['right'].set_visible(False)
# # ax2.spines['top'].set_visible(False)
# # ax3.spines['right'].set_visible(False)
# # ax3.spines['top'].set_visible(False)
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

