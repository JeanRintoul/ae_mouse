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
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
# 
savepath    = 'D:\\ae_mouse\\e121_stimulation\\t6_anesthesia\\'
# difference frequency. #
# data_type = '2low_anesthesia_t6'
data_type   = '2high_anesthesia_t6'

start_section  = 0.0
end_section    = 6.0

# save out at 1e4
dfx        = 1 
# low anesthesia, files that are bad. 
# dfx_files  = [51,52,53,54,55,56,57,58,59,60]
# high anesthesia, bad files are: 62, 63, 65, 68, 69
dfx_files = [61,64,66,67,70]
# 
# pulse data. 
# dfx_files = [31,32,33,34,35,36,37,38,39,40]
# 
# # 
print ('dfx files', dfx_files)
print ('frequency', dfx)
# 
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
# lfp_heights   = []
# lfp_heights2  = []
carrier_datas = []
final_peak_heights = []
r2s          = []
d2s          = []
lfp_peak_locations = []
rf_peak_locations  = []
raw_datas    = []
for i in dfx_files:     # 
    print ('i',i)
    file_number = i
    print ('file_number',file_number)
    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    data        = np.load(filename)
    fsignal     = 1e6*data[m_channel]/gain
    rfsignal    = 10*data[rf_channel]  
    # 
    # fft_start       = int(start_section*Fs)
    # fft_end         = int(end_section*Fs) 
    # fft_data        = fft(fsignal[fft_start:fft_end])
    # N               = fft_end - fft_start
    # fft_data        = np.abs(2.0/(N-1) * (fft_data))[1:(N-1)//2]
    # xf              = np.fft.fftfreq( (N-1), d=timestep)[:(N-1)//2]
    # frequencies     = xf[1:(N-1)//2]
    # 

    # bandwidth            = 1
    # df_idx               = find_nearest(frequencies,dfx)
    # lfp_height           = np.max(np.round(2*fft_data[df_idx-bandwidth:df_idx + bandwidth],2))
    # print ('LFP height', lfp_height)
    # 
    # dfx band filter. An alternate peak-peak measure. 
    bandwidth = 300
    bl = dfx - bandwidth
    if (dfx-bandwidth) <= 0:
        bl = 0.1
    bh = dfx + bandwidth
    sos_dfx_band = iirfilter(17, [bl,bh], rs=60, btype='bandpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    sos_carrier_band = iirfilter(17, [1e6-1,1e6 +2], rs=60, btype='bandpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    carrier_data              = sosfiltfilt(sos_carrier_band, fsignal)
    carrier_heights       = np.max(carrier_data) - np.min(carrier_data)    

    dfx_data              = sosfiltfilt(sos_dfx_band, fsignal)

    # Hilbert transform the results. 

    rfanalytical_signal  = hilbert(rfsignal)
    rfamplitude_envelope = np.abs(rfanalytical_signal)
    sos_lp = iirfilter(17, [1500], rs=60, btype='lowpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    rfhilberted             = sosfiltfilt(sos_lp, rfamplitude_envelope)
    rawdata                 = sosfiltfilt(sos_lp, fsignal)

    # decimate the data so it plots faster. 
    new_Fs                        = 1e4 
    downsampling_factor           = int(Fs/new_Fs)
    dt           = t[::downsampling_factor]
    dfx_data     = dfx_data[::downsampling_factor]
    rfh_data     = rfhilberted[::downsampling_factor]
    raw_data     = rawdata[::downsampling_factor]  

    peaksr, _   = find_peaks(rfh_data, prominence=20)
    npeaksr, _  = find_peaks(-rfh_data, prominence=20)    
    # print ('peaksr: ', peaksr)
    # print ('npeaksr: ', npeaksr)
    # 
    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(211)
    plt.plot(dt,dfx_data,'k')
    plt.plot(dt[peaksr],dfx_data[peaksr],'.r')    
    plt.plot(dt[npeaksr],dfx_data[npeaksr],'.c')    
    ax2  = fig.add_subplot(212)
    plt.plot(dt,rfh_data,'r')
    plt.plot(dt[peaksr],rfh_data[peaksr],'.k')    
    plt.plot(dt[npeaksr],rfh_data[npeaksr],'.c') 
    plt.show()

    # we select just the peaks where the DC offsets in the data are not.     
    max_indices = [0,2,4]
    min_indices = [0,2,3]
    # 
    band_search = int(0.2*new_Fs)     # 0.2 seconds on either side of the RF peak. 
    # print ('peaksr:', peaksr[max_indices])   

    # print ('npeaksr:', npeaksr[min_indices])
    peak_heights = []
    max_idxs = []
    for j in range(len(max_indices)):
        # print ('j',j,max_indices[j])
        min_number  = peaksr[max_indices[j]] - band_search
        max_number  = peaksr[max_indices[j]] + band_search
        nmin_number = npeaksr[min_indices[j]] - band_search
        nmax_number = npeaksr[min_indices[j]] + band_search 

        max_peak = np.max(dfx_data[min_number:max_number])
        min_peak = np.min(dfx_data[nmin_number:nmax_number])

        # get the indexes. 
        max_idx = find_nearest(dfx_data,max_peak)

        print ('max_idx:',max_idx)
        max_idxs.append(max_idx)
        peak_height = max_peak - min_peak
        # print ('peak height',peak_height)
        peak_heights.append(peak_height)
    #         
    print ('peaks r:',peaksr[max_indices])
    print ('peaks lfp:',max_idxs)

    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(211)
    plt.plot(dt,dfx_data,'k')
    plt.plot(dt[peaksr],dfx_data[peaksr],'.r')    
    plt.plot(dt[npeaksr],dfx_data[npeaksr],'.c')    
    ax2  = fig.add_subplot(212)
    plt.plot(dt,rfh_data,'r')
    plt.plot(dt[max_idxs],dfx_data[max_idxs],'.k')    
    plt.plot(dt[max_idxs],dfx_data[max_idxs],'.c') 
    plt.show()



    # use the RF peaks as Ground Truth. 
    for n in range(len(peak_heights)):
        final_peak_heights.append(peak_heights[n])
    # print ('len stuff: ',len(r2),len(d2))
    raw_datas.append(raw_data)
    carrier_datas.append(carrier_heights)
    r2s.append(rfh_data)                   # this is the middle section, which is aligned with the peak indices. 
    d2s.append(dfx_data)                   # 
    lfp_peak_locations.append(max_idxs)
    rf_peak_locations.append(peaksr[max_indices]) 
# 
# 
outfile                      = data_type+'_data.npz'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,lfp_peak_locations=lfp_peak_locations,rf_peak_locations=rf_peak_locations,final_peak_heights=final_peak_heights,carrier_datas=carrier_datas,r2s=r2s,d2s=d2s,raw_datas=raw_datas)
print ('saved out a data file!')

