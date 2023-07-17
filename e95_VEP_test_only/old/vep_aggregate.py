'''

Title: Aggregate the VEPS. Also implements a cross-correlation time based epoch removal based on a pearson cross-correlation. 

Author: Jean Rintoul
Date: 21.02.2023

'''
# 
# 
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
import pandas as pd
import sys
sys.path.append('D:\\mouse_aeti')  #  so that we can import from the parent folder. 
import mouse_library as m
# 
# 
start_files             = 1  #  pressure connected. 14hz
end_files               = 10  #  pressure connected. 14hz
files                   = np.linspace(start_files,end_files,(end_files-start_files+1),dtype=np.int16)
# 
files_to_average      = files
data_filepath         = 'D:\\mouse_aeti\\e95_VEP_test_only\\t4\\'
factor                = 1  # total cycles. 
gain                  = 1
led_frequency         = 7 # in Hz. 
single_file_duration  = 8.0
carrier_f             = 500000
duration              = single_file_duration 
print ('duration',duration)
led_duration          = 1/(2*led_frequency)
# when the led turned on and off. 
Fs                    = 5e6
timestep              = 1.0/Fs
N                     = int(Fs*duration)
# print("expected no. samples:",N)
marker_channel          = 7 
m_channel               = 0 
rf_channel              = 2

i_channel             = 5 
v_channel             = 6 

# create time and frequencies arrays.
t  = np.linspace(0, duration, N, endpoint=False)
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]
# 
low  = 1
high = 1000
sos_band = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

# decimation parameters. 
new_Fs = 10000 
downsampling_factor  = int(Fs/new_Fs)
print ('downsampling_factor',downsampling_factor)
# 
# This filter is to compare epochs with the base epoch. 
low  = 1
high = 100
vep_matcher_band = iirfilter(17, [low,high], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
epoch_array           = []
marker_array          = []
good_count = 0 
for i in range(len(files_to_average)):
  # Read in each file.  
  file_number       = files_to_average[i]
  print ('filenumber:',file_number)
  filename          = data_filepath + 't'+str(file_number)+'_stream.npy'
  data              = np.load(filename)
  a,b               = data.shape
  # 
  markerdata        = data[marker_channel]
  marker            = markerdata/np.max(markerdata)
  diffs             = np.diff( markerdata )
  zarray            = t*[0]
  indexes           = np.argwhere(diffs > 0.2)[:,0] 
  # indexes           = np.argwhere(diffs < -2.0)[:,0] 
  rawdata           = 1e6*data[m_channel]/gain
  filtered_rawdata  = rawdata
  # rfdata     = 10*data[rf_channel]
  mains_harmonics = [50,100,150]
  for i in range(len(mains_harmonics)):
      mains_low  = mains_harmonics[i] -2
      mains_high = mains_harmonics[i] +2
      mains_sos  = iirfilter(17, [mains_low,mains_high], rs=60, btype='bandstop',
                         analog=False, ftype='cheby2', fs=Fs,
                         output='sos')
      filtered_rawdata  = sosfiltfilt(mains_sos, filtered_rawdata)
  filtermain_rawdata    = sosfiltfilt(sos_band, filtered_rawdata)
  filtered_rawdata    = sosfiltfilt(vep_matcher_band, filtered_rawdata)

  print ('len indexes', len(indexes),indexes)
  factor_duration = int(2*led_duration * Fs *factor)
  # print ('factor duration',factor_duration)
  time_segment        = np.linspace(0, timestep*factor_duration, factor_duration, endpoint=False)
  # Read in the baseline flash. 
  filename          = 'chosen_vep_f1.npz'
  data              = np.load(filename)
  chosen_VEP        = data['chosen_VEP']
  chosen_marker     = data['chosen_marker']
  chosen_VEP = chosen_VEP - np.mean(chosen_VEP)
  # Epoch removal. 
  # Now make an inner loop to compare each LED flash to the baseline flash. 
  for j in range(len(indexes)-1):

    print ('index selection: ', (indexes[j+1]-int(3*factor_duration)/4))

    epoch = filtered_rawdata[int((indexes[j+1]-int(3*factor_duration)/4)):int((indexes[j+1]+int(factor_duration/4)) )]
    marker_epoch = markerdata[int((indexes[j+1]-int(3*factor_duration)/4)):int((indexes[j+1]+int(factor_duration/4)) )]
    main_epoch = filtermain_rawdata[int((indexes[j+1]-int(3*factor_duration)/4)):int((indexes[j+1]+int(factor_duration/4)) )]
    # subtract the mean first. 
    epoch = epoch-np.mean(epoch)
    df = pd.DataFrame({'x': np.real(epoch), 'y': np.real(chosen_VEP) })
    window = len(epoch)
    window          = int(len(epoch)/10 ) # this is the number of samples to be used in the rolling cross-correlation. 
    rolling_corr    = df['x'].rolling(window).corr(df['y'])
    # Really I just want a high correlation at the part of the file where there is the VEP.     
    median_corr = np.nanmedian(rolling_corr[1500000:2000000])
    print ('median correlation:',median_corr)
    # downsample the results. Note: main is already low pass filtered at 1khz. 
    main_epoch    = main_epoch[::downsampling_factor]
    marker_epoch  = marker_epoch[::downsampling_factor]
    time_seg      = time_segment[::downsampling_factor]
    # 
    epoch_array.append(main_epoch)
    marker_array.append(marker_epoch)
    good_count = good_count + 1
    # 
    # Epoch threshold decision. If the end part vaguely seems to correlate, we will call it good.  
    # if median_corr > 0.7: 
    #   print ('good one!')
    #   epoch_array.append(main_epoch)
    #   marker_array.append(marker_epoch)
    #   good_count = good_count + 1
    
    # Plot it out.  
    # fig = plt.figure(figsize=(10,6))
    # ax  = fig.add_subplot(211)
    # plt.plot(time_segment,epoch,'b')
    # plt.plot(time_segment,chosen_VEP,'r')
    # plt.plot(time_segment,20*chosen_marker,'g')
    # ax2  = fig.add_subplot(212)
    # plt.plot(rolling_corr,'g')
    # plt.show()
    # 
    # What I could have done is decimated the epoch_array before saving it out. 
# Now save out each epoch into an array. 
# That way later I can plot them all out as single instances. 
print ('number of good epochs: ',good_count)
# outfile="aggregated_veps.npz"   # save out the data. 
outfile="aggregated_noled_veps.npz"   # save out the data. 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile, epochs=np.array(epoch_array),markers=np.array(marker_array),time_segment=time_seg)
print ('saved out aggregated veps!')


# 
# END! 
# 
