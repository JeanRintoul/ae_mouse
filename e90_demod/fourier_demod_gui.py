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


start_files             = 43  #  no pressure connected. 14 hz
end_files               = 53  #  no pressure connected. 14hz

# start_files             = 20  #  pressure connected. 14hz
# end_files               = 31  #  pressure connected. 14hz, file 29 is good. 

# start_files             = 32  #  no pressure connected. 10hz
# end_files               = 42  #  no pressure connected. 10hz

data_filepath           = 'D:\\mouse_aeti\\e86_demod\\t10_mouse\\'
# Instead of events, if this is called it will just continuously loop until it finishes. Great if I want to go away and do something else. 
automated               = False
files                   = np.linspace(start_files,end_files,(end_files-start_files+1),dtype=np.int16)
# gains                   = 1*np.ones(len(files))
# f = files 
# g = gains 
# print ('all files: ',f)
# file 11,20 needs time syncing. 
gain                    = 1 
carrier_f               = 500000
led_frequency           = 14 # in Hz. 
duration                = 8.0 # time in S. 
# print ('duration',duration)
led_duration            = 1/(2*led_frequency)
Fs                      = 5e6
timestep                = 1.0/Fs
N                       = int(Fs*duration)
marker_channel          = 7 
m_channel               = 4 
rf_channel              = 2
t                       = np.linspace(0, duration, N, endpoint=False)
# xf                      = np.fft.fftfreq(N, d=timestep)[:N//2]
# frequencies             = xf[1:N//2]

low                     = 2
high                    = 50
sos_band                = iirfilter(17, [low,high], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')
# 
fc = carrier_f
# remove the original VEP signal from the data such that it doesn't interfere with the demodulation. 
l = fc - 2*high 
h = fc + 2*high
sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


newN                    = int(5*Fs)
xf                      = np.fft.fftfreq(newN, d=timestep)[:newN//2]
frequencies             = xf[1:newN//2]
# Now calculate the FFT over each entire file. 
window = np.hanning(newN)

idx                             = 0 
fft_rawdata               = np.zeros(len(frequencies))
fft_demoddata             = np.zeros(len(frequencies))
# IQ demodulate function. 
# I use constant to get rid of the rectification around zero issue. 
def demodulate(measured_signal,carrier_f,t):
    # That's interesting, if I add a DC offset here, I obtain the result correctly, otherwise the result is rectified into obscurity. 
    offset = 100
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

def update_fft(x):
    global fft_demoddata,fft_rawdata

    file_number   = files[x]
    print ('current file:',str(file_number))
    filename    = data_filepath + 't'+str(file_number)+'_stream.npy'
    data = np.load(filename)
    a,b = data.shape
    print ('shape',a,b)
    rawdata               = np.array(1e6*data[m_channel]/gain)
    rawdata               = rawdata - np.mean(rawdata)
    rfdata                = data[rf_channel]
    marker                = data[marker_channel]/np.max(data[marker_channel])
    diffs   = np.diff( marker)
    zarray  = t*[0]
    indexes = np.argwhere(diffs > 0.5)[:,0] # leading edge
    print ('first and last index',indexes[1],indexes[len(indexes)-1])
    start_idx = int(indexes[1])
    end_idx   = int(start_idx + 5*Fs)

    # before demodulating, ensure none of the original signal enters into the multiplication.  
    fraw                    = sosfiltfilt(sos_carrier_band, rawdata) 
    demodulated_signal      = demodulate(fraw,carrier_f,t)
    # filter the raw data the same way as the demod signal is filtered. 
    rawdata     = sosfiltfilt(sos_band, rawdata)
    demoddata   = sosfiltfilt(sos_band, demodulated_signal)

    # Use a hanning window to reduce spectal leakage issue. 
    fft_rawdata = fft(rawdata[start_idx:end_idx]*window)
    fft_rawdata = np.abs(2.0/newN * (fft_rawdata))[1:newN//2]

    fft_demoddata = fft(demoddata[start_idx:end_idx]*window)
    fft_demoddata = np.abs(2.0/newN * (fft_demoddata))[1:newN//2]

    # total_fft_rawdata   = np.add(total_fft_rawdata,fft_rawdata)
    # total_fft_demoddata = np.add(total_fft_demoddata,fft_demoddata)

    # results = [total_fft_rawdata,total_fft_demoddata]  
    results = [fft_rawdata, fft_demoddata]  
    return results

# 
fig = plt.figure(figsize=(10,7),num ='Fourier Analysis of Demodulated Data')
ax = fig.add_subplot(211)
fig.subplots_adjust(bottom=0.15) 
plt.plot(frequencies,fft_demoddata,'purple')
ax.set_xlim([0,high])
ax.set_xlabel('Frequency(Hz)')
ax.set_ylabel('Volts ($\mu$V)')

ax2 = fig.add_subplot(212)
plt.plot(frequencies,fft_rawdata,'k')
ax2.set_xlim([0,high])
ax2.set_xlabel('Frequency(Hz)')
ax2.set_ylabel('Volts ($\mu$V)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

# setup a slider axis and the Slider
ax_depth = plt.axes([0.09, 0.02, 0.56, 0.04])
slider_depth = Slider(ax_depth, 'file idx: ', 0, len(files)-1, valinit=round(idx),valfmt='%d' )
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
    # print ('filenumber',f[idx],idx)
    print ('fft bin size:', frequencies[5]-frequencies[4])
    results = update_fft(idx)    

    if ((idx > (len(files)-2) ) and automated) or not automated: # only update the plots at the end if automated, otherwise update every single time. 
        # update the line graphs. 
        # trawfft     = total_fft_rawdata/(idx)   - fft_brawdata
        # tdemodfft   = total_fft_demoddata/(idx) - fft_bdemoddata
        trawfft     = fft_rawdata/(idx)   
        tdemodfft   = fft_demoddata/(idx)       
        # print ('line subtracted datas', len(fft_brawdata),len(trawfft))
        # fft bdemod data, looks exactly like the top demod data. i.e. the baseline and the actual data look the same?  
        # in fact the baseline is a bit larger in amplitude than the averaged one.  
        ax.cla()
        ax2.cla()
        # ax.plot(frequencies,total_fft_rawdata/(idx),'r')
        # ax.plot(frequencies,total_fft_demoddata/(idx),'purple')
        # ax.plot(frequencies,trawfft,'orange')
        ax.plot(frequencies,tdemodfft,'purple')        
        ax.set_xlim([0,high])
        # ax.set_ylim([0,5])        
        ax.set_ylabel('Volts ($\mu$V)')
        ax.set_xlabel('Frequency(Hz)')
        # ax2.plot(frequencies,total_fft_demoddata/(idx+1),'purple')
        # ax2.plot(frequencies,fft_brawdata,'k')
        # ax2.plot(frequencies,total_fft_rawdata/(idx),'k')
        ax2.plot(frequencies,trawfft,'k')
        print ('idx:',idx)
        # ax2.plot(frequencies,fft_bdemoddata,'r')                
        ax2.set_xlim([0,high])
        # ax2.set_ylim([0,10])       
        ax2.set_ylabel('Volts ($\mu$V)')   
        ax2.set_xlabel('Frequency(Hz)')

        fig.canvas.draw_idle()


def on_press(event):
    print('press', event.key)
    sys.stdout.flush()    
    if event.key == 'l' or event.key=='right' and idx < (len(files)-1):        
        print ('idx',idx)
        # idx = idx + 1     
        num = idx + 1   
        print ('filenumber',files[num]) 
        slider_depth.set_val( int(num) )
        # 
        # Problem with this is it doesn't allow time to render.  
        if automated is True:
            while idx < (len(files)-1):
                num = idx + 1
                print ('filenumber',files[num]) 
                slider_depth.set_val( int(num) )
    else: 
        print ('you reached the end of the file list. ')
# 
fig.canvas.mpl_connect('key_press_event', on_press)
slider_depth.on_changed(update_plot)
plt.show()









