'''

Title: Aggregate VEP height data for violin plot over multiple files. 

Author: Jean Rintoul
Date: 19.06.23

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
# 
# 
# acoustically connected. 
start_files     = 5
end_files       = 30
# 
# not acoustically connected. 
# start_files     = 46
# end_files       = 70
# 
savepath        = 'D:\\mouse_aeti\\e97_MEPS\\t18_mouse_eeg\\'
# 
# file_number     = 67
# 
file_list       = np.linspace(start_files,end_files,(end_files-start_files+1),dtype=np.int16)
print ('file list:',file_list)

gain            = 1000
led_frequency   = 4 # in Hz. The number of times the LED is on per second. 
factor          = 1
Fs              = 5e6
duration        = 8.0
timestep        = 1.0/Fs
N               = int(Fs*duration)
marker_channel  = 7 
i_channel       = 5
v_channel       = 6   
m_channel       = 0     

led_duration    = 1/(2*led_frequency) # when the led turned on and off. 
start_time      = 0.0
end_time        = start_time+led_duration
start           = factor*led_duration   # this is where the trial starts seconds before marker. 
end             = factor*led_duration*2   # this is where the trial ends
low  = 0.5
high = 1000 
high = 50 
sos_low = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

veps = np.zeros((1875000))

height_list = []
for i in range(len(file_list)):
    file_number   = file_list[i]
    print ('current file:',str(file_number))
    filename    = savepath + 't'+str(file_number)+'_stream.npy'
    data = np.load(filename)
    a,b = data.shape
    t = np.linspace(0, duration, N, endpoint=False)
    # convert it to microvolts by taking the gain into account. 
    fsignal = 1e6*data[m_channel]/gain
    # do some mains filtering. 
    mains_harmonics = [50,100,150,200,250,300]
    # mains_harmonics = [50,100]
    for j in range(len(mains_harmonics)):
        mains_low  = mains_harmonics[j] -2
        mains_high = mains_harmonics[j] +2
        mains_sos = iirfilter(17, [mains_low,mains_high], rs=60, btype='bandstop',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
        fsignal = sosfiltfilt(mains_sos, fsignal)
    # remove anything above 1000Hz.  
    fsignal  = sosfiltfilt(sos_low, fsignal)
    # create the marker channel. 
    markerdata        = np.array(data[marker_channel])
    marker            = markerdata/np.max(markerdata)
    diffs             = np.diff( markerdata )
    zarray            = t*[0]
    indexes           = np.argwhere(diffs > 0.2)[:,0] # leading edge
    marker_up_length  = int(led_duration*Fs)
    for k in range(len(indexes)):
        if k > 0:
            zarray[indexes[k]:(indexes[k]+marker_up_length) ] = 1
    markerdata = zarray
    # Get the individual events. 
    pre_event_idxcount              = int(start*Fs)
    post_event_idxcount             = int(end*Fs)
    data_to_segment                 = fsignal
    filtered_segmented_data         = []
    for m in range(len(indexes)-1):  # the last index may get chopped short. 
      filtered_segmented_data.append(data_to_segment[(indexes[m+1]-pre_event_idxcount):(indexes[m+1]+post_event_idxcount)])
    # 
    filtered_segmented_array = np.array(filtered_segmented_data)
    a,b = filtered_segmented_array.shape
    print ('shape', a,b)
    time_segment    = np.linspace(-start,end,num=b)
    # average_lfp     = np.median(filtered_segmented_array,axis=0)
    # std_lfp         = np.std(filtered_segmented_array,axis=0)
    print ('n_events: ', a)
    # 
    # I am seeking the min and max between two points during this 4Hz induced VEP. 
    seg_vep_start = 0.03
    seg_vep_end   = 0.09
    vep_start_idx = find_nearest(time_segment, seg_vep_start)
    vep_end_idx   = find_nearest(time_segment, seg_vep_end)
    # 
    vep_area = filtered_segmented_array[:,vep_start_idx:vep_end_idx]
    # append all the information together, when vep height above some base threshold. 
    for j in range(a): 
        vheight = np.max(vep_area[j,:]) - np.min(vep_area[j,:]) 
        if vheight >= 0:     # only take veps 50 microvolts or greater. s
            height_list.append(vheight)
            v_one = filtered_segmented_array[j,:]
            veps = np.vstack( (veps, v_one) )
            print ('veps shape: ',veps.shape)
# 
print ('len heightlist:', len(height_list),height_list)
print ('veps:', veps.shape)
# 
outfile        = 'D:\\mouse_aeti\\e97_MEPS\\mouse_eeg.npz'
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,height_list=height_list, veps=veps,time_segment=time_segment)
print ('saved out a data file!')

