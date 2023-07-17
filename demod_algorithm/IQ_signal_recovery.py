'''

Title: IQ demodulation realdata
Function: Perform IQ demodulation. 
This function has a rolling window pearson cross-correlation. 

Author: Jean Rintoul
Date: 18.03.2022

Updated: 22.05.2022


'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import kaiserord
from scipy import signal
from scipy.signal import iirfilter,sosfilt
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
#
# I still have the rectification problem in the demodulation result. 
#
def demodulate(measured_signal,carrier_f):
    # That's interesting, if I add a DC offset here, I obtain the result correctly, otherwise the result is rectified into obscurity. 
    # adding an offset changes the signals recovered size.. 
    offset = 0
    offset_adjustment = offset*np.sin(2 * np.pi * carrier_f * t)
    IQ = (measured_signal+offset_adjustment)*np.exp(1j*(2*np.pi*carrier_f*t )) 
    idown = np.real(IQ)
    qdown = np.imag(IQ)
    # idown = measured_signal*np.cos(2*np.pi*carrier_f*t)
    # qdown = -measured_signal*np.sin(2*np.pi*carrier_f*t)    
    v = idown + 1j*qdown
    # calculate the phase in order to figure out the sign of the recovered wave. 
    mag     = np.abs(IQ)  # magnitude
    # phase   = np.angle(v) # phase angle
    # instantaneous phase. 
    # phase   = np.arctan(qdown/idown)

    phase = np.unwrap(np.angle(v))#inst phase
    inst_freq = np.diff(phase)/(2*np.pi)*Fs #inst frequency
    # Regenerate the carrier from the instantaneous phase
    # regenerated_carrier = np.cos(inst_phase)
    # #
    # mag = mag - np.mean(mag)
    return mag,phase


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx    


#  these are the channels in the data file that are used: uren phantom
# uren phantom
# filename = 'uren_demod_stream.npy'
# rf_channel          = 0    #  this is the rf amplifier output monitor. 
# v_channel           = 4    #  this is output voltage monitor after David Bono's voltage transformer i.e. the voltage being applied across the stimulation electrodes. This measurement uses a 10x attenuator so I don't draw any of the power away. 
# ae_channel          = 6    #  this is measuring the acoustoelectric voltage i.e. it goes through a hardware high pass filter before being amplified. 
# vmeasure_channel    = 5    #  this is measured voltage before the hardware filter. It is measured with a 10x attenuator so as not to impact the ampliture of the signal going through the hardware filter. 
# duration            = 0.1   # uren phantom duration
# start_slice         = 0.03
# end_slice           = 0.08

# mouse saline phantom. 
# filename = 'mouse_saline_phantom.npy'
# ae_channel   = 0
# rf_channel   = 2
# vmeasure_channel    = 5
# duration = 0.1   # uren phantom duration
# duration = 8.0  #  mouse saline phantom duration
# start_slice  = 4.8
# end_slice    = 5.0


# mouse saline phantom. 
filename = 't7_stream.npy'
ae_channel   = 6
rf_channel   = 2
vmeasure_channel    = 6
# duration = 0.1   # uren phantom duration
duration = 4.0  #  mouse saline phantom duration
start_slice  = 2.2
end_slice    = 2.4

Fs       = 5e6     # sample rate
timestep = 1.0/Fs
N = int(Fs*duration)
print("expected no. samples:",N)
t = np.linspace(0, duration, N, endpoint=False)
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]



data = np.load(filename)
print (data.shape)
fft_vmfield = fft(10*data[vmeasure_channel])
fft_vm = fft(10*data[vmeasure_channel])

carrier_f               = 500000
filter_cutoff           = 20000


lp_filter                = iirfilter(17, [filter_cutoff], rs=60, btype='lowpass',
                            analog=False, ftype='cheby2', fs=Fs,
                            output='sos')


raw_demodulated_signal,phase  = demodulate(data[ae_channel],carrier_f)

demodulated_signal    = sosfiltfilt(lp_filter, raw_demodulated_signal)
# raw_v_at_electrode = sosfiltfilt(lp_filter, 10*data[vmeasure_channel])
raw_v_at_electrode = sosfiltfilt(lp_filter, data[ae_channel])



# # remove the starting ramp and ending ramp. 
val,idx_start   = find_nearest(t,start_slice)
val,idx_end     = find_nearest(t,end_slice)

# clip the start and end areas away where the current signal is ramping. 
raw_v_at_electrode = raw_v_at_electrode[idx_start:idx_end]
demodulated_signal = demodulated_signal[idx_start:idx_end]
t_cut              = t[idx_start:idx_end]
phase               = phase[idx_start:idx_end]
# 
# Correctly conpute the lag between the two signals. 
x = raw_v_at_electrode
y = demodulated_signal
correlation = signal.correlate(x-np.mean(x), y - np.mean(y), mode="full")
lags = signal.correlation_lags(len(x), len(y), mode="full")
idx_lag = -lags[np.argmax(correlation)]
print ('lag is',idx_lag)
#
if idx_lag > 0:
    raw_v_at_electrode = raw_v_at_electrode[:(len(demodulated_signal)-idx_lag)]
    t_cut              = t_cut[idx_lag:]
    demodulated_signal = demodulated_signal[idx_lag:]
else: 
    raw_v_at_electrode = raw_v_at_electrode[-idx_lag:]
    t_cut              = t_cut[-idx_lag:]
    demodulated_signal = demodulated_signal[:(len(demodulated_signal)+idx_lag)]
#  
# demodulated_signal = demodulated_signal-np.mean(demodulated_signal)
raw_v_at_electrode = raw_v_at_electrode-np.mean(raw_v_at_electrode)

df = pd.DataFrame({'x': np.real(raw_v_at_electrode), 'y': np.real(demodulated_signal) })
window          = 12000 #this is the number of samples to be used in the rolling cross-correlation. 
# window          = len(demodulated_signal)
rolling_corr    = df['x'].rolling(window).corr(df['y'])

print ('rolling corr max:',np.max(rolling_corr))  # 
result = np.nanmedian(rolling_corr)
print ('median corr: ',result)
max_index = np.argmax(rolling_corr) 


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(3,1,1)

# plt.plot(np.real(raw_v_at_electrode),'r')

plt.plot(raw_v_at_electrode,'r')
plt.plot(demodulated_signal,'black')

plt.legend(['raw v at electrode','demodulated signal'])
ax2 = fig.add_subplot(3,1,2)
plt.plot(phase,'purple')

ax3 = fig.add_subplot(3,1,3)
plt.plot(t_cut,np.real(abs(rolling_corr)),'.b')
plt.legend(['abs(correlation) window:'+str(window)])
plt.show()


# # # # #  Plots for paper. # #  #  #  #  
# # These are the "Tableau Color Blind 10" colors as RGB.    
# tableaucolorblind10 = [
# (0, 107, 164), 
# (255, 128, 14), 
# (171, 171, 171), 
# (89, 89, 89),    
# (95, 158, 209), 
# (200, 82, 0), 
# (137, 137, 137), 
# (162, 200, 236),  
# (255, 188, 121), 
# (207, 207, 207)]    
  
# # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
# for i in range(len(tableaucolorblind10)):    
#     r, g, b = tableaucolorblind10[i]    
#     tableaucolorblind10[i] = (r / 255., g / 255., b / 255.) 
# plt.rcParams['axes.linewidth'] = 2

# start_pt = 0.03
# end_pt =    0.0325

# save_title = 'ionic_input_signal'
# fig = plt.figure(figsize=(7,7))
# ax = fig.add_subplot(1,1,1)
# plt.plot(t_cut,np.real(raw_v_at_electrode),color = tableaucolorblind10[3],linewidth=2.0 )

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
# ax.set_xlim([start_pt,end_pt])
# plt.savefig(save_title+".png",transparent=True, pad_inches=0,bbox_inches='tight')
# plt.show()




# save_title = 'demodulated_signal'
# fig = plt.figure(figsize=(7,7))
# ax = fig.add_subplot(1,1,1)
# plt.plot(t_cut,np.real(demodulated_signal),color = tableaucolorblind10[3] ,linewidth=2.0)
# ax.set_xlim([start_pt,end_pt])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
# plt.savefig(save_title+".png",transparent=True, pad_inches=0,bbox_inches='tight')
# plt.show()

# d_lowcut  = 480000 # 504000
# d_highcut = 520000 # 514000
# sos_demod = signal.iirfilter(17, [d_lowcut, d_highcut], rs=60, btype='band',
#                        analog=False, ftype='cheby2', fs=Fs,
#                        output='sos')
# ae_demod_input = signal.sosfilt(sos_demod, data[ae_channel])

# save_title = 'heterodyned_signal'
# fig = plt.figure(figsize=(7,7))
# ax = fig.add_subplot(1,1,1)
# plt.plot(t,ae_demod_input,color = tableaucolorblind10[3],linewidth=2.0 )
 
# plt.axis('off')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.xticks([])
# plt.yticks([])
# ax.set_xlim([start_pt,end_pt])
# plt.savefig(save_title+".png",transparent=True, pad_inches=0,bbox_inches='tight')
# plt.show()