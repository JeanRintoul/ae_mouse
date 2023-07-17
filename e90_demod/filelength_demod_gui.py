"""

Look in fourier space at the entire file length for each recorded file. 

Display the 0-60Hz region
Directly below show the carrier_f, carrier_f + 60 

What I could do is a better baseline subtraction. 
i.e. take a baseline with no led. do the FFT, and subtract it from... 

"""
import sys
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.widgets import TextBox
from scipy import signal
from scipy.signal import iirfilter,sosfiltfilt
from scipy.fft import fft,fftshift
import pandas as pd
# 
start_files             = 21  #  pressure connected. 14hz
end_files               = 31  #  pressure connected. 14hz
# 
# start_files             = 43  #  no pressure connected. 14 hz
# end_files               = 53  #  no pressure connected. 14hz
# 
data_filepath           = 'D:\\mouse_aeti\\e86_demod\\t10_mouse\\'

# start_files             = 5  #  pressure connected. 14hz
# end_files               = 14  #  pressure connected. 14hz
# data_filepath           = 'D:\\mouse_aeti\\e86_demod\\t8\\'
# good files: good_file_list = [21,23,27]
# what is weird is the recovered data is larger than the data at the base freq. 

mains_filter            = False # runs faster without the mains filter - dont really need for high N. 

# Instead of events, if this is called it will just continuously loop until it finishes. Great if I want to go away and do something else. 
automated        = False
files            = np.linspace(start_files,end_files,(end_files-start_files+1),dtype=np.int16)
gains            = 1*np.ones(len(files))
f = files 
g = gains 
# print ('all files: ',f)
# file 11,20 needs time syncing. 
carrier_f               = 500000
# carrier_f             = 672800
led_frequency           = 14 # in Hz. 
# t10
duration                = 8.0 # time in S. 
Fs                      = 5e6
# t8
# Fs                      = 1e7
# duration                = 3.0 # time in S. 
gain                    = 1
# print ('duration',duration)
led_duration            = 1/(2*led_frequency)
timestep                = 1.0/Fs
N                       = int(Fs*duration)
# t10
marker_channel          = 7 
m_channel               = 4 
rf_channel              = 2

#t8
# marker_channel          = 3 
# m_channel               = 0 
# rf_channel              = 2


t                       = np.linspace(0, duration, N, endpoint=False)

start_pause = 2 
end_pause   = 7

# start_pause = 1 
# end_pause   = 2.5

# if i go too low, i can see dc drifts too much. 
low                     = 4.5
high                    = 50

low  = 11 
high = 17 
sos_band                = iirfilter(17, [low,high], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')

glow                     = 13
ghigh                    = 15
sos_bandstop                = iirfilter(17, [glow,ghigh], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')


fc = carrier_f
# remove the original VEP signal from the data such that it doesn't interfere with the demodulation. 
l = fc - 2*high 
h = fc + 2*high
sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


# decimation parameters. 
# new_Fs                        = 50000 
# downsampling_factor           = int(Fs/new_Fs)
start_idx = int(start_pause*Fs)
end_idx   = int(end_pause*Fs)
fft_start       = int(start_idx) 
fft_end         = int(end_idx)
newN            = int(fft_end - fft_start)
window          = np.hanning(newN)
xf              = np.fft.fftfreq(newN, d=timestep)[:newN//2]
frequencies     = xf[1:newN//2]
carrier_signal   = np.sin(2 * np.pi * carrier_f * t)

idx                             = 0 
rawdata                         = np.zeros(N)
demodulated_signal              = np.zeros(N)
marker                          = np.zeros(N)
rfdata                          = np.zeros(N)
rolling_corr                    = np.zeros(N)
fft_rawdata                     = np.zeros(len(frequencies))
fft_demoddata                   = np.zeros(len(frequencies))
fft_rfdemoddata                 = np.zeros(len(frequencies))

# IQ demodulate function. 
# I use constant to get rid of the rectification around zero issue. 
def demodulate(measured_signal,carrier_f,t):
    # That's interesting, if I add a DC offset here, I obtain the result correctly, otherwise the result is rectified into obscurity. 
    offset = 10000
    offset_adjustment = offset*np.sin(2 * np.pi * carrier_f * t)
    IQ = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
    # idown = measured_signal*np.cos(2*np.pi*carrier_f*t)
    # qdown = -measured_signal*np.sin(2*np.pi*carrier_f*t)    
    idown = np.real(IQ)
    qdown = np.imag(IQ)
    v = idown + 1j*qdown
    mag = np.abs(v)
    mag = mag - np.mean(mag)
    return mag
# 
def update_file(x):
    global rawdata,demodulated_signal,marker,rfdata,fft_demoddata,fft_rawdata,rolling_corr,fft_rfdemoddata

    file_number   = f[x]
    print ('current file:',str(file_number))
    filename    = data_filepath + 't'+str(file_number)+'_stream.npy'
    data = np.load(filename)
    a,b = data.shape
    print ('shape',a,b)
    markerdata          = data[marker_channel]
    marker              = markerdata/np.max(markerdata)
    diffs   = np.diff( markerdata )
    zarray  = t*[0]
    indexes = np.argwhere(diffs > 0.2)[:,0] # leading edge
    marker_up_length = int(led_duration*Fs)
    for i in range(len(indexes)):
        if i > 0:
            zarray[indexes[i]:(indexes[i]+marker_up_length) ] = 1
    marker = zarray

    rfdata              = 10*data[rf_channel]    
    # rfdemodulated       = rfdata*carrier_signal 

    rawdata             = 1e6*data[m_channel]/gain
    rawdata             = rawdata-np.mean(rawdata)
    #    
    fraw                = sosfiltfilt(sos_carrier_band, rawdata) 
    demodulated_signal  = demodulate(fraw,carrier_f,t)
    #  both signals are filtered the same way. 
    demodulated_signal  = sosfiltfilt(sos_band, demodulated_signal) 
    rawdata             = sosfiltfilt(sos_band, rawdata) 

    # demodulated_signal  = sosfiltfilt(sos_bandstop, demodulated_signal) 
    # rawdata             = sosfiltfilt(sos_bandstop, rawdata) 

    df = pd.DataFrame({'x': rawdata-np.mean(rawdata), 'y': demodulated_signal-np.mean(demodulated_signal) })
    cwindow          = 12000*10
    # window         = len(demodulated_signal)
    rolling_corr     = df['x'].rolling(cwindow).corr(df['y'])
    print ('newN',newN,len(rawdata[fft_start:fft_end]))
    fft_rawdata = fft(rawdata[fft_start:fft_end]*window)
    fft_rawdata = np.abs(2.0/newN * (fft_rawdata))[1:newN//2]

    fft_demoddata = fft(demodulated_signal[fft_start:fft_end]*window)
    fft_demoddata = np.abs(2.0/newN * (fft_demoddata))[1:newN//2]

    # fft_rfdemoddata = fft(rfdemodulated[fft_start:fft_end]*window)
    # fft_rfdemoddata = np.abs(2.0/newN * (fft_rfdemoddata))[1:newN//2]
    # print ('len fft rf',len(fft_rfdemoddata))
    results = [rawdata,demodulated_signal,marker,rfdata,fft_demoddata,fft_rawdata,rolling_corr,fft_rfdemoddata]
    return results

print ('len of ffts',len(frequencies),len(fft_rawdata))
# 
fig = plt.figure(figsize=(10,7),num ='Looking at the whole file')
ax = fig.add_subplot(311)
fig.subplots_adjust(bottom=0.15) 
plt.plot(t[fft_start:fft_end],rawdata[fft_start:fft_end],'k')
plt.plot(t[fft_start:fft_end],demodulated_signal[fft_start:fft_end],'r')
plt.plot(t[fft_start:fft_end],50*marker[fft_start:fft_end],'g')
ax.set_ylabel('Volts ($\mu$V)')

ax2 = fig.add_subplot(312)
# plt.plot(t,rolling_corr,'g')
plt.plot(frequencies,fft_rawdata,'k')
plt.plot(frequencies,fft_demoddata,'r')
# plt.plot(frequencies,fft_rfdemoddata,'b')
plt.legend(['raw','demod'],loc='upper right')
ax2.set_xlim([0,high])

ax3 = fig.add_subplot(313)
# plt.plot(t[fft_start:fft_end],marker[fft_start:fft_end],'b')
plt.plot(rolling_corr,'k')
# plt.plot(t,total_demoddata,'k')
# plt.plot(t,total_markerdata,'g')
# # ax2.set_xlim([0,high])
# # ax2.set_xlabel('Frequency(Hz)')
# ax2.set_ylabel('Volts ($\mu$V)')
# 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
# setup a slider axis and the Slider
ax_depth = plt.axes([0.09, 0.02, 0.56, 0.04])
slider_depth = Slider(ax_depth, 'file idx: ', 0, len(f), valinit=round(idx),valfmt='%d' )
ax_depth.axis('off')

# axbox = fig.add_axes([0.8, 0.02, 0.1, 0.04])
# text_box = TextBox(axbox, "N flashes: ", textalignment="center")
# # text_box.on_submit(submit)
# text_box.set_val("0")  # Trigger `submit` with the initial string.

# update the figure when button is pressed.  
def update_plot(val):
    global idx

    # outfile="rawdlfpata.npz"   # save out the data. 
    # np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
    # np.savez(outfile, total_average_lfp=total_average_lfp, total_demod_average_lfp=total_demod_average_lfp)
    # print ('saved out a data file!')

    idx = int(round(slider_depth.val))
    results = update_file(idx)    

    print (len(t),len(rolling_corr))
    if ((idx > (len(f)-2) ) and automated) or not automated: # only update the plots at the end if automated, otherwise update every single time. 

        ax.cla()
        ax2.cla()

        ax.plot(t[fft_start:fft_end],demodulated_signal[fft_start:fft_end],'r')   
        ax.plot(t[fft_start:fft_end],rawdata[fft_start:fft_end],'k')        
        ax.plot(t[fft_start:fft_end],20*marker[fft_start:fft_end],'g')                 
        ax.set_xlim([start_pause,end_pause])
        ax.set_ylabel('Volts ($\mu$V)')

        # ax2.plot(t,rolling_corr,'k')
        ax2.plot(frequencies,fft_rawdata,'k')
        ax2.plot(frequencies,fft_demoddata,'r')   
        # ax2.plot(frequencies,fft_rfdemoddata,'b')
        ax2.set_xlim([0,high])     
        plt.legend(['raw','demod'],loc='upper right')
        ax3.plot(t,rolling_corr,'k')
        # ax3.plot(t[fft_start:fft_end],rfdata[fft_start:fft_end],'b')  
        ax3.set_xlim([start_pause,end_pause])

        fig.canvas.draw_idle()
# when loading, start with the first file. 
update_plot(0)

def on_press(event):
    print('press', event.key)
    sys.stdout.flush()    
    if event.key == 'l' or event.key=='right' and idx < (len(f)-1):        
        print ('idx',idx)
        # idx = idx + 1     
        num = idx + 1   
        print ('filenumber',f[num]) 
        slider_depth.set_val( int(num) )
        # 
        # Problem with this is it doesn't allow time to render.  
        if automated is True:
            while idx < (len(f)-1):
                num = idx + 1
                print ('filenumber',f[num]) 
                slider_depth.set_val( int(num) )
    else: 
        print ('you reached the end of the file list. ')
# 
fig.canvas.mpl_connect('key_press_event', on_press)
slider_depth.on_changed(update_plot)
plt.show()









