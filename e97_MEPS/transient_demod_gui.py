"""
There appeaer to be sometime discontinuities at the location of the demod VEP.
Even when gain is 1000. What is happening here? Maybe my sample rate is too low for 672khz? 

Maybe there is some strange aliasing effect?
I could: 
1. increase sample rate. 
2. change to lower frequency transducer. i.e. 500kHz. 
3. add a filter that starts at Fs/2 to minimize this aliasing effect. Its possible that higher frequency harmonics are causing this step deviation. 

This also needs to cycle through in an automated fashion. 

"""
import sys
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib
import matplotlib.cm as cm
from scipy import interpolate
from numpy.linalg import norm
# from scipy.misc import derivative
# from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.signal import hilbert, chirp
from tkinter import *
import matplotlib.colors as colors
from matplotlib.widgets import TextBox
from scipy import signal
from scipy.signal import iirfilter,sosfiltfilt
from scipy.fft import fft,fftshift
#  
#  
factor                  = 1
carrier_f               = 500000
led_frequency           = 4 # in Hz. 

# start_files             = 3  #  pressure connected. 14hz
# end_files               = 26  #  pressure connected. 14hz
start_files             = 6 #  no pressure connected. 14 hz
end_files               = 30  #  no pressure connected. 14hz


data_filepath           = 'D:\\mouse_aeti\\e97_MEPS\\t18_mouse_eeg\\'
#  
mains_filter            = False # runs faster without the mains filter - dont really need for high N. 
# Instead of events, if this is called it will just continuously loop until it finishes. Great if I want to go away and do something else. 
automated               = False
files                   = np.linspace(start_files,end_files,(end_files-start_files+1),dtype=np.int16)
# good_file_list = [20,21,23,27]

gain = 500 
# files = good_file_list
gains                   = gain*np.ones(len(files))
f = files 
g = gains 
# print ('all files: ',f)
# file 11,20 needs time syncing. 

duration                = 8.0 # time in S. 
# print ('duration',duration)
led_duration            = 1/(2*led_frequency)
Fs                      = 5e6
timestep                = 1.0/Fs
N                       = int(Fs*duration)

marker_channel          = 7 
m_channel               = 0 
rf_channel              = 2
t                       = np.linspace(0, duration, N, endpoint=False)
xf                      = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies             = xf[1:N//2]

# low  = led_frequency - 1
# high = led_frequency + 1 
# low  = 2 # 4
# high = 50


low = 1
high = 1000 
high = 140 
sos_band                = iirfilter(17, [low,high], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')

fc = carrier_f
# remove the original VEP signal from the data such that it doesn't interfere with the demodulation. 
# remove the original VEP signal from the data such that it doesn't interfere with the demodulation. 
l = fc - high 
h = fc + high
sos_carrier_band = iirfilter(17, [l,h], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


sos_center_cut = iirfilter(17, [carrier_f-2,carrier_f+2], rs=60, btype='bandstop',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

filter_cutoff           = high
lp_filter               = iirfilter(17, [filter_cutoff], rs=60, btype='lowpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')



# decimation parameters. 
new_Fs                        = 50000 
downsampling_factor           = int(Fs/new_Fs)

# These are the variables which we wish to obtain at the end. 
factor_duration                 = int(led_duration*Fs*factor)
time_segment                    = np.linspace(0, timestep*factor_duration*2, factor_duration*2, endpoint=False) 
time_segment                    = time_segment - led_duration
# print ('factor length:', factor_duration)
idx                             = 0 
total_average_lfp               = np.zeros((factor_duration*2))
total_av_marker_check           = np.zeros((factor_duration*2))
total_std_lfp                   = np.zeros((factor_duration*2))
total_demod_average_lfp         = np.zeros((factor_duration*2))
total_demod_std_lfp             = np.zeros((factor_duration*2))
newN                            = int(factor_duration*2)
xf                              = np.fft.fftfreq(newN, d=timestep)[:newN//2]
newfrequencies                  = xf[1:newN//2]
total_fft_rawdata               = np.zeros(len(newfrequencies))
total_fft_demoddata             = np.zeros(len(newfrequencies))
total_number_of_events          = 0 

# IQ demodulate function. 
# I use constant to get rid of the rectification around zero issue. 
# def demodulate(measured_signal,carrier_f,t):
#     # That's interesting, if I add a DC offset here, I obtain the result correctly, otherwise the result is rectified into obscurity. 
#     offset = 100
#     offset_adjustment = offset*np.sin(2 * np.pi * carrier_f * t)
#     IQ = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
#     # idown = measured_signal*np.cos(2*np.pi*carrier_f*t)
#     # qdown = -measured_signal*np.sin(2*np.pi*carrier_f*t)    
#     idown = np.real(IQ)
#     qdown = np.imag(IQ)
#     v = idown + 1j*qdown
#     mag = np.abs(v)
#     mag = mag - np.mean(mag)
#     return mag


def demodulate(measured_signal,carrier_f,t):
    offset              = np.min(measured_signal)
    offset_adjustment   = offset*np.cos(2 * np.pi * carrier_f * t)
    IQ                  = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
    idown               = np.real(IQ)
    qdown               = np.imag(IQ)
    idown               = sosfiltfilt(lp_filter, idown)
    qdown               = sosfiltfilt(lp_filter, qdown)  
    rsignal             = (idown + qdown)
    rsignal             = rsignal - np.mean(rsignal) 
    return rsignal



def update_VEP(x):
    global total_average_lfp,total_number_of_events,total_av_marker_check,total_std_lfp,total_demod_average_lfp,total_demod_std_lfp,total_fft_demoddata,total_fft_rawdata

    file_number   = f[x]
    print ('current file:',str(file_number))
    filename    = data_filepath + 't'+str(file_number)+'_stream.npy'
    data = np.load(filename)
    a,b = data.shape
    print ('shape',a,b)
    rawdata                 = np.array(1e6*data[m_channel]/g[x])
    rawdata                 = rawdata - np.mean(rawdata)
    rfdata                  = 10*data[rf_channel]
    # this comes out with similar frequencies, and looking like the inverse 180 degree phase shift of the data. 
    fraw                    = sosfiltfilt(sos_carrier_band, rawdata) 
    fraw                    = sosfiltfilt(sos_center_cut, fraw) 
       
    demodulated_signal      = demodulate(fraw,carrier_f,t)
    # filter the raw data the same way as the demod signal is filtered. 
    rawdata                 = sosfiltfilt(sos_band, rawdata)
    demodulated_signal      = sosfiltfilt(sos_band, demodulated_signal)
    markerdata        = np.array(data[marker_channel])
    # end downsampling. 
    marker  = markerdata/np.max(markerdata)

    marker                  = markerdata/np.max(markerdata)
    diffs   = np.diff( markerdata )
    zarray  = t*[0]
    indexes = np.argwhere(diffs > 0.2)[:,0] # leading edge
    marker_up_length = int(led_duration*Fs)
    for i in range(len(indexes)):
        if i > 0:
            zarray[indexes[i]:(indexes[i]+marker_up_length) ] = 1
    markerdata = zarray

    print ('LED indexes length and positions: ',len(indexes),indexes)
    # carrier_signal = np.sin(2 * np.pi * carrier_f * t)
    rd = np.zeros((factor_duration*2))
    dd = np.zeros((factor_duration*2))
    mm = np.zeros((factor_duration*2))
    rf = np.zeros((factor_duration*2))

    good_events = 0 
    for i in range(len(indexes)):  #  go to second last index... as we skip the first index. 
        if i > 0 and i%factor == 0 : # skip the first marker. 
            startid = indexes[i] - int(factor_duration/(2*factor) ) 
            stopid  = indexes[i] + int(2*factor_duration - int(factor_duration/(2*factor)  ))
            print ('start and stop markers',startid,stopid)
            # this is baselien subtraction when its a single VEP. 
            # if factor == 1: 
            prestimulus_period          = rawdata[startid:indexes[i]]
            demod_prestimulus_period    = demodulated_signal[startid:indexes[i]]
            baseline                    = np.mean(prestimulus_period)
            demod_baseline              = np.mean(demod_prestimulus_period)
            # else:                           # for multiple VEPs. 
            #     baseline        = 0
            #     demod_baseline  = 0            
            # It's here we need to make a decision about the rf data. 
            rsample      = rawdata[startid:stopid]- baseline
            demod_sample = demodulated_signal[startid:stopid]-demod_baseline
            # Epoch removal.  
            # Now we want to eliminate flashes where the pressure changed suddenly. 
            d = np.diff(demod_sample[0::1000])
            startpt = 0
            thresholder = abs(np.max(np.abs(d[startpt:])))
            print ('threshold:',thresholder,len(d),len(time_segment) )

            # fig = plt.figure(figsize=(10,6))
            # ax  = fig.add_subplot(211)
            # plt.plot(time_segment,rsample,'b')
            # plt.plot(time_segment,demod_sample,'k')
            # # 
            # ax  = fig.add_subplot(212) 
            # plt.plot(d,'r')        
            # plt.suptitle(str(thresholder))
            # plt.show()
            # 
            # What happens if I incorporate the VEP quality measure in here? 

            # Only add the values where there are no discontinuities in the pressure gradient. 
            # if thresholder < 0.2: 
            if thresholder < 100:                 
                rd = np.add(rd,rsample) 
                dd = np.add(dd,demod_sample)
                mm = np.add(mm,markerdata[startid:stopid] )
                rf = np.add(rf,rfdata[startid:stopid] )            
                # print ('rd: ',len(rd))
                good_events = good_events + 1
            else:
                print ('flash eliminated')
                # fig = plt.figure(figsize=(10,6))
                # ax  = fig.add_subplot(211)
                # plt.plot(time_segment,rsample,'b')
                # plt.plot(time_segment,demod_sample,'k')                
                # ax  = fig.add_subplot(212) 
                # plt.plot(d,'r')     
                # plt.suptitle(str(thresholder))   
                # plt.show()

    # n_events = len(indexes)-1
    # print ('n_events: ', len(indexes)-1)
    n_events = good_events
    # print ('n_events: ', n_events)
    if n_events > 0:
        mm = mm/n_events
        dd = dd/n_events
        rd = rd/n_events
        rf = rf/n_events

        average_lfp         = rd
        av_marker_check     = mm
        std_lfp             = np.std(rd,axis=0)
        demod_average_lfp   = dd
        demod_std_lfp       = np.std(dd,axis=0)

        # print ('sshape: ',average_lfp.shape)
        # fig = plt.figure(figsize=(10,6))
        # ax  = fig.add_subplot(111)
        # plt.plot(time_segment,average_lfp,'r')
        # plt.show()

        # Total number of events so far.  
        total_number_of_events        = total_number_of_events + n_events
        total_average_lfp             = np.add(total_average_lfp,average_lfp)
        total_av_marker_check         = np.add(total_av_marker_check,av_marker_check)
        total_std_lfp                 = np.add(total_std_lfp ,std_lfp)
        total_demod_average_lfp       = np.add(total_demod_average_lfp,demod_average_lfp)
        total_demod_std_lfp           = np.add(total_demod_std_lfp,demod_std_lfp)
        #  
        # Now calculate the FFT. 
        newN = int(factor_duration*2)
        window = np.hanning(newN)

        fft_rawdata = fft(total_average_lfp*window)
        fft_rawdata = np.abs(2.0/newN * (fft_rawdata))[1:newN//2]

        fft_demoddata = fft(total_demod_average_lfp*window)
        fft_demoddata = np.abs(2.0/newN * (fft_demoddata))[1:newN//2]

        xf = np.fft.fftfreq(newN, d=timestep)[:newN//2]
        newfrequencies = xf[1:newN//2]

        total_fft_rawdata = np.add(total_fft_rawdata,fft_rawdata)
        total_fft_demoddata = np.add(total_fft_demoddata,fft_demoddata)

    results = [total_number_of_events, total_average_lfp, total_av_marker_check,total_std_lfp,total_demod_average_lfp, total_demod_std_lfp,time_segment,total_fft_rawdata,total_fft_demoddata ]
    
    return results



# 
fig = plt.figure(figsize=(10,7),num ='Visual Evoked Potentials')
ax = fig.add_subplot(221)
fig.subplots_adjust(bottom=0.15) 
ax.plot(time_segment,total_average_lfp/(idx+1), 'r')
ax.plot(time_segment,total_demod_average_lfp/(idx+1), 'purple')
ax.plot(time_segment, total_av_marker_check/(idx+1) ,'g')
ax.set_xlim([np.min(time_segment),np.max(time_segment)])
ax.title.set_text('VEP')
ax.set_xlabel('Time(s)')
ax.set_ylabel('Volts ($\mu$V)')

ax2 = fig.add_subplot(222)
ax2.plot(time_segment,total_demod_average_lfp/(idx+1), 'purple')
ax2.plot(time_segment, total_av_marker_check/(idx+1) ,'g')
ax2.title.set_text('demodulated VEP')
ax2.set_xlim([np.min(time_segment),np.max(time_segment)])
ax2.set_xlabel('Time(s)')

ax3 = fig.add_subplot(223)
plt.plot(newfrequencies,total_fft_rawdata)
ax3.set_xlim([0,high])
ax3.set_xlabel('Frequency(Hz)')
ax3.set_ylabel('Volts ($\mu$V)')

ax4 = fig.add_subplot(224)
plt.plot(newfrequencies,total_fft_demoddata)
ax4.set_xlim([0,high])
ax4.set_xlabel('Frequency(Hz)')


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

# setup a slider axis and the Slider
ax_depth = plt.axes([0.09, 0.02, 0.56, 0.04])
slider_depth = Slider(ax_depth, 'file idx: ', 0, len(f)-1, valinit=round(idx),valfmt='%d' )
ax_depth.axis('off')

axbox = fig.add_axes([0.8, 0.02, 0.1, 0.04])
text_box = TextBox(axbox, "N flashes: ", textalignment="center")
# text_box.on_submit(submit)
text_box.set_val("0")  # Trigger `submit` with the initial string.


# update the figure when button is pressed.  
def update_plot(val):
    global idx

    outfile="rawdlfpata.npz"   # save out the data. 
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
    np.savez(outfile, total_average_lfp=total_average_lfp, total_demod_average_lfp=total_demod_average_lfp)
    print ('saved out a data file!')


    print ('fft bin size:', newfrequencies[5]-newfrequencies[4])

    idx = int(round(slider_depth.val))
    # print ('filenumber',f[idx],idx)
    results = update_VEP(idx)
    # print ('no of events from results',results[0])

    print ('total number of events:', total_number_of_events)

    if ((idx > (len(f)-2) ) and automated) or not automated: # only update the plots at the end if automated, otherwise update every single time. 
        # update the line graphs. 
        ax.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()

        # scale the lfp. 
        max_height  = np.max(total_average_lfp/(idx+1))/2
        m_height    = (total_av_marker_check/(idx+1) )
        max_marker_height = np.max((total_av_marker_check/(idx+1) ))
        ax.plot(time_segment, max_height*m_height/max_marker_height ,'g')
        # ax.plot(time_segment, total_av_marker_check/(idx+1) ,'g')
        ax.plot(time_segment,total_average_lfp/(idx+1), 'r')
        # uncomment below:
        ax.plot(time_segment,total_demod_average_lfp/(idx+1), 'purple')

        ax.set_xlabel('Time(s)')
        ax.set_xlim([np.min(time_segment),np.max(time_segment)])
        ax.set_ylabel('Volts ($\mu$V)')
        ax.title.set_text('VEP')
        ax2.plot(time_segment,total_demod_average_lfp/(idx+1), 'purple')
        # scale the lfp. 
        max_height  = np.max(total_demod_average_lfp/(idx+1))/2
        m_height    = (total_av_marker_check/(idx+1) )
        max_marker_height = np.max((total_av_marker_check/(idx+1) ))

        print ('demod av lfp height',max_marker_height, max_height)    
        ax2.plot(time_segment, max_height*m_height/max_marker_height ,'g')
        ax2.set_xlabel('Time(s)')
        ax2.set_xlim([0,np.max(time_segment)])
        ax2.set_xlim([np.min(time_segment),np.max(time_segment)])  
        ax2.title.set_text('demodulated VEP')

        ax3.plot(newfrequencies,total_fft_rawdata/(idx+1),'r')
        ax3.plot(newfrequencies,total_fft_demoddata/(idx+1),'purple')
        ax3.set_xlim([0,high])
        ax3.set_ylabel('Volts ($\mu$V)')
        ax3.set_xlabel('Frequency(Hz)')

        ax4.plot(newfrequencies,total_fft_demoddata/(idx+1),'purple')
        ax4.set_xlim([0,high])
        ax4.set_xlabel('Frequency(Hz)')

        fig.canvas.draw_idle()
        text_box.set_val(str(total_number_of_events))  # Trigger `submit` with the initial string.


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



