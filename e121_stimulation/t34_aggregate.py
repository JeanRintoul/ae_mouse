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
# savepath    = 'D:\\ae_mouse\\e121_stimulation\\t4_phantom\\'
savepath    = 'D:\\ae_mouse\\e121_stimulation\\t3_mouse\\'
# difference frequency. 
data_type = 'mouse'

start_section  = 1.7
end_section    = 5.5

# save out at 1e4
# dfx         = 1 
# dfx_files  = range(1,21)
# dfx         = 2
# dfx_files  = range(21,35)
dfx        = 5
dfx_files  = range(36,51)
# dfx        = 10
# dfx_files  = range(51,61)
# dfx        = 40
# dfx_files  = range(61,63)
# dfx        = 100
# dfx_files  = range(63,65)
# dfx        = 300
# dfx_files  = range(65,67)
# I only care about amplitude, and not phase after this.
# save out at a higher sample rate Fs = 1e6, that way I can plot the phase, but I do not have accurate peaks. 
# dfx        = 1000
# dfx_files  = range(67,69)
# dfx        = 2000
# dfx_files  = range(69,71)
# dfx        = 5000
# dfx_files  = range(71,73)
# 
print ('dfx files', dfx_files)
print ('frequency',dfx)
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
lfp_heights  = []
r2s          = []
d2s          = []
peakrs       = []
peakds       = []
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
    fft_start       = int(start_section*Fs)
    fft_end         = int(end_section*Fs) 
    fft_data        = fft(fsignal[fft_start:fft_end])
    N               = fft_end - fft_start
    fft_data        = np.abs(2.0/(N-1) * (fft_data))[1:(N-1)//2]
    xf              = np.fft.fftfreq( (N-1), d=timestep)[:(N-1)//2]
    frequencies     = xf[1:(N-1)//2]
    # 
    # fig = plt.figure(figsize=(10,6))
    # ax = fig.add_subplot(211)
    # plt.plot(t,fsignal,'k')
    # ax2 = fig.add_subplot(212)
    # plt.plot(frequencies,fft_data,'k')
    # ax2.set_xlim([0,dfx+50])
    # plt.show()
    # 
    bandwidth            = 1
    df_idx               = find_nearest(frequencies,dfx)
    lfp_height           = np.max(np.round(2*fft_data[df_idx-bandwidth:df_idx + bandwidth],2))
    print ('LFP height', lfp_height)
    # 
    # dfx band filter. An alternate peak-peak measure. 
    bandwidth = 1
    bl = dfx - bandwidth
    if (dfx-bandwidth) <= 0:
        bl = 0.1
    bh = dfx + bandwidth
    sos_dfx_band = iirfilter(17, [bl,bh], rs=60, btype='bandpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    dfx_data              = sosfiltfilt(sos_dfx_band, fsignal)

    # Hilbert transform the results. 
    try:
        rfanalytical_signal  = hilbert(rfsignal)
    except: 
        print ('memory allocation error trying again')
        rfanalytical_signal  = hilbert(rfsignal)

    rfamplitude_envelope = np.abs(rfanalytical_signal)
    sos_lp = iirfilter(17, [dfx*2+100], rs=60, btype='lowpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    rfhilberted             = sosfiltfilt(sos_lp, rfamplitude_envelope)
    rawdata                 = sosfiltfilt(sos_lp, fsignal)
    # decimate the data so it plots faster. 
    new_Fs                        = 1e4
    if dfx > 100:
        new_Fs                  = 1e6     
    downsampling_factor           = int(Fs/new_Fs)
    dt           = t[::downsampling_factor]
    dfx_data     = dfx_data[::downsampling_factor]
    rfh_data     = rfhilberted[::downsampling_factor]
    raw_data     = rawdata[::downsampling_factor]    
    start_idx   = find_nearest(dt,start_section)
    end_idx     = find_nearest(dt,end_section)
    r2          = rfh_data[start_idx:end_idx]
    d2          = dfx_data[start_idx:end_idx]
    t2          = dt[start_idx:end_idx]
    # dfx = 1 
    # peaksr, _   = find_peaks(r2, prominence=10)
    # peaksd, _   = find_peaks(d2, prominence=10)
    peaksr, _   = find_peaks(r2, prominence=5)
    peaksd, _   = find_peaks(d2, prominence=5)    
    # peaksr, _ = find_peaks(r2, prominence=1)
    # peaksd, _ = find_peaks(d2, prominence=1)    
    print ('lengths peaksr/peaksd:', len(peaksr),len(peaksd) )
    print ('peaksr: ',peaksr)
    print ('peaksd: ',peaksd)
    # 
    # just for dfx = 2 
    # if (len(peaksr) > len(peaksd) and len(peaksr) == 8 ):
    #     print ('in the special exception for dfx = 2')
    #     peaksr = peaksr[0:7]
    # just for dfx = 1 
    # if (len(peaksr) > len(peaksd) and len(peaksr) == 4 and len(peaksd)==2):
    #     print ('in the special exception for dfx = 1')
    #     peaksr = peaksr[1:3]
    #     fig = plt.figure(figsize=(10,6))
    #     ax = fig.add_subplot(111)
    #     plt.plot(peaksr,r2[peaksr],'or')
    #     plt.plot(r2,'r')
    #     plt.plot(peaksd,d2[peaksd],'ob')
    #     plt.plot(d2,'b')
    #     plt.show()
    # 
    # just for dfx = 5    
    if (len(peaksr) > len(peaksd) and len(peaksr) == 19 and len(peaksd)==18):
        print ('in the special exception for dfx = 5')
        peaksr = peaksr[0:18]
    
    # just for dfx = 10   
    # if (len(peaksr) > len(peaksd) and len(peaksr) == 37 and len(peaksd)==36):
    #     print ('in the special exception for dfx = 10')
    #     peaksr = peaksr[0:36]
    # 
    # n = int(input("Choose to manually edit: "))
    # if n==1: 
    #     peaksr = np.append(peaksr,1) 
    #     print ('peaksr: ',peaksr)
    #     print ('peaksd: ',peaksd)
    # 
    # 
    # only futz with it if I have to. 
    if (len(peaksr) != len(peaksd) ):
    
        print ('diff is: ',len(peaksr)-len(peaksd))

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

    print ('lengths',len(peaksr),len(peaksd) )

    # if (len(peaksr) != len(peaksd) ):
    
    #     fig = plt.figure(figsize=(10,6))
    #     ax = fig.add_subplot(111)
    #     plt.plot(peaksr,r2[peaksr],'or')
    #     plt.plot(r2,'r')
    #     plt.plot(peaksd,d2[peaksd],'ob')
    #     plt.plot(d2,'b')
    #     plt.show()

    #     # number of elements
    #     n = int(input("Enter number of elements peaksr: "))
    #     if n > 0:
    #         if n == 9:
    #             # save out peaksr. 
    #             df = pd.DataFrame(peaksr)
    #             df.to_csv('data.csv', sep=',')               
    #             n = int(input("ready?: "))
    #             if n == 1: 
    #                 df = pd.read_csv('data.csv').copy()
    #                 result = np.array(df.values.tolist()).T
    #                 result =  result[1,:]
    #                 peaksr = []
    #                 for m in range(0, len(result)):
    #                     if result[m] !=' ':
    #                         peaksr.append(int(result[m]))
    #                 print ('newpeaksr: ',peaksr)
    #         else:
    #             # Below line read inputs from user using map() function
    #             peaksr = list(map(int, input("\nEnter correct peaksr : ").strip().split()))[:n]
    #             peaksr = np.array([int(p) for p in peaksr]) 
    #             print("\npeaksr is - ", peaksr)
    #     # 
    #     # number of elements
    #     n = int(input("Enter number of elements peaksd: "))
    #     if n > 0:
    #         if n == 9:
    #             # save out peaksd. 
    #             df = pd.DataFrame(peaksd)
    #             df.to_csv('data.csv', sep=',')               
    #             n = int(input("ready?: "))
    #             if n == 1: 
    #                 df = pd.read_csv('data.csv').copy()
    #                 result = np.array(df.values.tolist()).T
    #                 result =  result[1,:]
    #                 peaksd = []
    #                 for m in range(0, len(result)):
    #                     if result[m] !=' ':
    #                         peaksd.append(int(result[m]))
    #                 print ('newpeaksd: ',peaksd)
    #         else:
    #             # Below line read inputs from user using map() function
    #             peaksd = list(map(int, input("\nEnter correct peaksd : ").strip().split()))[:n]
    #             print("\npeaksd is - ", peaksd)

    print ('len stuff: ',len(r2),len(d2))
    raw_datas.append(raw_data)
    lfp_heights.append(lfp_height)   # this is the FFT height at the difference frequency. 
    r2s.append(r2)                   #  this is the middle section, which is aligned with the peak indices. 
    d2s.append(d2)                   # 
    # apparently a list of lists is the way to go here.
    peakrs.append(peaksr)
    peakds.append(peaksd)
# 
# 
outfile                      = 'f_'+str(dfx)+'Hz_'+data_type+'_data.npz'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,lfp_heights=lfp_heights,r2s=r2s,d2s=d2s,peakrs=peakrs,peakds=peakds,raw_datas=raw_datas)
print ('saved out a data file!')

