'''

Title: compare the data going into generator with the data coming out of generator. 

SNR analysis of t5. 
Author: Jean Rintoul
Date:   05.02.2024

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
# from scipy.signal import iirfilter,sosfiltfilt
# from math import floor
# from scipy.signal import hilbert
from scipy.stats import ttest_ind
# 
def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
# 
saveprefix          = './/images//'
# 

# Continuous wave. 
continuous_savepath            = 'D:\\ae_mouse\\e135_ae_neural_decoding\\t5_phantom\\g1_continuous_wave\\'
filename            = 't9_stream.npy'
continuous_file_list = [9,10,11,12]

# PRF Wave. 
prf_savepath            = 'D:\\ae_mouse\\e135_ae_neural_decoding\\t5_phantom\\g1_prf_with_pressure\\'
# filename            = 't1_stream.npy'
PRF_file_list = [1,2,3,4]


duration            = 12
Fs                  = 2e6
timestep            = 1.0/Fs
N                   = int(Fs*duration)
carrier_frequency   = 500000
prf                 = 80
channel_of_interest = 0
rfchannel           = 4
gain                = 100
t                   = np.linspace(0, duration, N, endpoint=False)
current_signal_frequency = 24
band_of_interest    = 40


print ('current_signal_frequency: ', current_signal_frequency)
start_time      = np.round(0.8/duration ,2)
end_time        = np.round((duration - 0.2)/duration,2)
print ('start and end',start_time,end_time)
start_pause     = int(start_time * N+1)
end_pause       = int(end_time *N-1)
xf = np.fft.fftfreq( (end_pause-start_pause), d=timestep)[:(end_pause-start_pause)//2]
frequencies = xf[1:(end_pause-start_pause)//2]

prf_snrs    = [] 
cont_snrs   = []
for n in range(4):
    prf_filename            = 't'+str(PRF_file_list[n])+'_stream.npy'
    continuous_filename     = 't'+str(continuous_file_list[n]) + '_stream.npy'
    print ('filename: ',n,prf_filename, continuous_filename)
    prf_data = np.load(prf_savepath+prf_filename)
    continuous_data = np.load(continuous_savepath+continuous_filename)
    # 
    continuous_signal   =  1e6*continuous_data[channel_of_interest]/gain
    prf_signal          =  1e6*prf_data[channel_of_interest]/gain
    # 
    fft_c = fft(continuous_signal[start_pause:end_pause])
    fft_c = np.abs(2.0/(end_pause-start_pause) * (fft_c))[1:(end_pause-start_pause)//2]
    # 
    fft_prf = fft(prf_signal[start_pause:end_pause])
    fft_prf = np.abs(2.0/(end_pause-start_pause) * (fft_prf))[1:(end_pause-start_pause)//2]
    # Now calculate the signal to noise ratios. 

    cont_start = find_nearest(frequencies,carrier_frequency-band_of_interest)
    cont_end   = find_nearest(frequencies,carrier_frequency+band_of_interest)

    cont_inner_start = find_nearest(frequencies,carrier_frequency-15)
    cont_inner_end   = find_nearest(frequencies,carrier_frequency+15)

    prf_start  = find_nearest(frequencies,prf-band_of_interest)
    prf_end    = find_nearest(frequencies,prf+band_of_interest)

    # Calculate PRF SNR. 
    prf_totals                   = []
    interest_frequencies         = []
    frequencies_of_interest = [prf-current_signal_frequency,prf+current_signal_frequency]
    for i in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
      df_idx = find_nearest(frequencies,frequencies_of_interest[i])         
      interest_frequencies.append(df_idx)    
      prf_totals.append(fft_prf[df_idx])
    dis_prf_totals = []  # sum all frequencies per unit time. 
    for i in range(prf_start,prf_end): # sum all frequencies per unit time.
      if i not in interest_frequencies: 
          dis_prf_totals.append(fft_prf[i])
    prf_snr   = 20*np.log(np.mean(prf_totals)/np.mean(dis_prf_totals))
    print ('prf snr:',prf_snr)

    # Calculate continuous SNR. 
    continuous_totals            = []
    interest_frequencies         = []
    frequencies_of_interest = [carrier_frequency-current_signal_frequency,carrier_frequency+current_signal_frequency]
    for i in range(len(frequencies_of_interest)): # sum all frequencies per unit time.
      df_idx = find_nearest(frequencies,frequencies_of_interest[i])         
      interest_frequencies.append(df_idx)    
      continuous_totals.append(fft_c[df_idx])
    dis_continuous_totals = []  # sum all frequencies per unit time. 
    for i in range(cont_start,cont_end): # sum all frequencies per unit time.
      if i not in interest_frequencies: 
        if (i < cont_inner_start or i > cont_inner_end):
          dis_continuous_totals.append(fft_c[i])
    continuous_snr   = 20*np.log(np.mean(continuous_totals)/np.mean(dis_continuous_totals))
    print ('continuous snr:',continuous_snr)
    # 
    prf_snrs.append(prf_snr)
    cont_snrs.append(continuous_snr)
#
print (prf_snrs)
print (cont_snrs)
#
# Now make a bar chart with standard deviation and T-test. 
# 
prf_mean  = np.mean(prf_snrs) 
prf_std   = np.std(prf_snrs) 
c_mean    = np.mean(cont_snrs) 
c_std     = np.std(cont_snrs) 


# T-test. 
sample1 = prf_snrs
sample2 = cont_snrs
t_stat, p_value = ttest_ind(sample1, sample2) 
print('T-statistic value: ', t_stat) 
print('P-Value: ', p_value)

# Create lists for the plot
materials = ['PRF', 'Continuous']
x_pos = np.arange(len(materials))
CTEs = [prf_mean,c_mean]
error = [prf_std,c_std]

print ('means:',CTEs)
print ('stds:',error)

# Build the plot
fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', ecolor='black', capsize=10)
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,color='grey', capsize=10)

ax.set_xticks(x_pos)
ax.set_xticklabels([])
# ax.yaxis.grid(True)

plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Save the figure and show
plt.tight_layout()
plt.savefig(saveprefix+'bar_plot_with_error_bars.png')
plt.show()
