import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq,fftshift,ifft,ifftshift
from scipy.signal import blackman
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.signal import fftconvolve
#
# Design a band pass filter to isolate just the information I want to get the peak to peak amplitude more accurately.
# 
# 

  # // Channel identities: 
  # // 1. rf amplifier output
  # // 2. hydrophone probe
  # // 

  # // 3. current monitor for e1
  # // 4. current monitor for e2

  # // 5. v mon e1 x10  
  # // 6. v mon e2 x10 

  # // 7. tiepie voltage output waveform x10 
  # // 8. diff input voltage across 1k resister. x 10
Fs = 5e6
duration = 3.0
N = Fs*duration
print("expected no. samples:",N)
t = np.linspace(0, duration, int(N), endpoint=False)
resistor_current_mon = 49.9  #  49.9 Ohms for current monitor, 1k resistor 

#  Filter Design 
nyq_rate = Fs / 2.0
# # The desired width of the transition from pass to stop,
# relative to the Nyquist rate.  
width = 500.0/nyq_rate
# The desired attenuation in the stop band, in dB.
ripple_db = 60.0
# Compute the order and Kaiser parameter for the FIR filter.
Ntaps, beta = kaiserord(ripple_db, width)
# The cutoff frequency of the filter.
lowcut  = 492000  -3000 # 504000
highcut = 492000 + 3000 #  514000
sos = signal.iirfilter(17, [lowcut, highcut], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
w, h = signal.sosfreqz(sos, 2000, fs=Fs)

e_lowcut  = 8000 - 1000
e_highcut = 8000 + 2000
sos_efield = signal.iirfilter(17, [e_lowcut, e_highcut], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
w_e, h_e = signal.sosfreqz(sos, 2000, fs=Fs)                       


ae_idx = 500000-8000
e_idx  = 8000

timestep = 1.0/Fs
N = int(N)
# w = blackman(int(N)) # use the entire data. 
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

element,ae_idx = find_nearest(frequencies,ae_idx)
# print ('ae idx is: ',element,ae_idx)
element,e_idx = find_nearest(frequencies,e_idx)
# print ('e idx is: ',element,e_idx)

rf_channel  = 0    
hydrophone_channel = 1 
i_channel   = 3
v_channel   = 4 
ae_channel  = 6  
e_channel   = 7 

resistor_current_mon = 49.9 
E_gap          = 0.0025
AE_gap         = 0.0014
gain           = 100


filename = 'old_ati_stream_data.npy'

d = np.load(filename)
a,b,c = d.shape
data = d.transpose(1,0,2).reshape(b,-1) 
# print (data.shape)

fft_aefield = fft(data[ae_channel])
fft_ae = np.abs(2.0/N * (fft_aefield))[1:N//2]
# ae_V_pp_fft = fft_ae[ae_idx]*2
# AE_V_x = ae_V_pp_fft 
# AE_E_x = (AE_V_x/AE_gap)/gain
# print ('AE V_x (V) AE E_x (V/m): ',AE_V_x,AE_E_x)
ae_filtered = signal.sosfilt(sos, data[ae_channel])
# ae_voltage_pp = np.pi*sum(np.fabs(ae_filtered))/len(ae_filtered)
# AE_scale = (ae_voltage_pp/AE_gap)/gain
# print ('AE V_x filtered (V) AE E_x filter (V/m)',ae_voltage_pp,(ae_voltage_pp/AE_gap)/gain )

fft_efield = fft(10*data[e_channel])
fft_e = np.abs(2.0/N * (fft_efield))[1:N//2]
# V_pp_fft = fft_e[e_idx]*2
# E_x = V_pp_fft/AE_gap
# print ('V_x(V) E_x(V/m): ',V_pp_fft,E_x)
e_filtered = signal.sosfilt(sos_efield, 10*data[e_channel])
# V_x_scale = np.pi*sum(np.fabs(e_filtered))/len(e_filtered)
# E_x_scale = V_x_scale/AE_gap
# print ('V_x(V) scale E_x(V/m) scale:',V_x_scale,E_x_scale) 
print ('-----------MAX and RMS-----------------')
# Max AE and ratio
aemax_pp_filtered = np.max(ae_filtered) - np.min(ae_filtered)
AE_max = (aemax_pp_filtered/AE_gap)/gain
E_max = (np.max(e_filtered)-np.min(e_filtered))/AE_gap
print ('AE_filtered max and E max(V/m)',AE_max,E_max)
print ('E/AE max ratio: ',E_max/AE_max)

#  RMS AE Amplitude Calculation
AE_rms = (np.sqrt(np.mean(ae_filtered**2))/AE_gap)/gain
E_rms=np.sqrt(np.mean(e_filtered**2))/AE_gap  
print ('AE_filtered and E rms(V/m)',AE_rms,E_rms)
print ('E/AE RMS ratio',E_rms/AE_rms)

# ratio = E_x/AE_E_x
# print ('applied e to ae ratio: ',ratio)
print ('------------PROBE V,I,R----------------')
V     = 10*data[v_channel]
I     = 5*data[i_channel]/resistor_current_mon
V_pp  = np.pi*sum(np.fabs(V))/len(V)
I_pp  = np.pi*sum(np.fabs(I))/len(I)
impedance = np.divide(V,I, where=I!=0)
impedance = np.abs(np.median(impedance))
print ('V_pp, I_pp(mA),R',V_pp, I_pp*1000,impedance)

# fft_rf = fft(data[rf_channel])
# fft_rf = np.abs(2.0/N * (fft_rf))[1:N//2]

# fft_eappliedfield = fft(10*data[v_channel])
# fft_eappliedfield = np.abs(2.0/N * (fft_eappliedfield))[1:N//2]

# fft_ifield = fft(5*data[i_channel]/resistor_current_mon)
# fft_ifield = np.abs(2.0/N * (fft_ifield))[1:N//2]
# These are the "Tableau Color Blind 10" colors as RGB.    
tableaucolorblind10 = [
(0, 107, 164), 
(255, 128, 14), 
(171, 171, 171), 
(89, 89, 89),    
(95, 158, 209), 
(200, 82, 0), 
(137, 137, 137), 
(162, 200, 236),  
(255, 188, 121), 
(207, 207, 207)]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableaucolorblind10)):    
    r, g, b = tableaucolorblind10[i]    
    tableaucolorblind10[i] = (r / 255., g / 255., b / 255.) 

fig = plt.figure(figsize=(8,5.5))
ax = fig.add_subplot(111)
plt.plot(t,data[e_channel],color = tableaucolorblind10[1])
plt.plot(t,data[ae_channel],color = tableaucolorblind10[0])
# plt.plot(t,ae_filtered,color = tableaucolorblind10[1])
ax.legend(['Applied V@measurement probe(V)','Raw AE(V)'],loc='upper left',framealpha= 0.0,fontsize = 14)
plt.xlim([0.4998,0.500])
plt.ylim([-0.3,0.35])

stepsize = 0.00005
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, stepsize))

plt.ylabel('AE Volts(V)',fontsize=14)
plt.xlabel('time(seconds)',fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("raw_mixed_signal.svg", bbox_inches="tight",format="svg") 
plt.show()


fig = plt.figure(figsize=(7,5.5))
ax = fig.add_subplot(111)
plt.plot(frequencies, fft_ae, color = tableaucolorblind10[0])
plt.xlim([0,1100000])
stepsize = 500000
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, stepsize))

ax.legend(['AE FFT(V)'],loc='upper left',framealpha= 0.0,fontsize=14)
plt.ylabel('Volts(V)',fontsize=14)
plt.xlabel('Frequency(Hz)',fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# a = plt.axes([0.18, 0.6, .25, .25]), facecolor=tableaucolorblind10[6]
a = plt.axes([.58, .6, .3, .3])
plt.plot(frequencies, fft_ae, color = tableaucolorblind10[1])
# plt.ylim([0,0.045])
plt.xlim(485000, 515000)
# plt.xticks([])
plt.yticks([])

plt.savefig("fft_mixed_signal.svg", bbox_inches="tight",format="svg") 
plt.show()



# fig = plt.figure(figsize=(10,7.5))
# ax = fig.add_subplot(3,1,1)
# plt.plot(t,V,'r')
# plt.plot(t,10*data[e_channel],color = tableaucolorblind10[4])
# plt.ylabel('Volts')
# ax.legend(['V','V@measureprobe'])

# ax2 = fig.add_subplot(3,1,2)
# plt.plot(t,data[ae_channel],'r')
# plt.plot(t,ae_filtered,'b')
# ax2.legend(['raw AE(V) through SR560','508khz band filter AE(V)'],loc='upper right')
# plt.xlim([0.5,0.5005])
# plt.ylabel('Volts')

# ax3 = fig.add_subplot(3,1,3)
# plt.plot(frequencies, fft_ae, '-b')
# plt.xlim([0,1100000])
# ax3.legend(['AE FFT(V)'],loc='upper right')
# plt.xlabel('Frequency(Hz)')
# plt.ylabel('V')
# # ax.spines['right'].set_visible(False)
# # ax.spines['top'].set_visible(False)
# # ax2.spines['right'].set_visible(False)
# # ax2.spines['top'].set_visible(False)
# plt.show()

