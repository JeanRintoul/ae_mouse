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
duration = 8.0
bandwidth = 4 
f = 8000
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
lowcut  = abs(500000-f)- bandwidth/2 # 504000
highcut = abs(500000-f) + bandwidth/2 #  514000
sos = signal.iirfilter(17, [lowcut, highcut], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# w, h = signal.sosfreqz(sos, 2000, fs=Fs)

lowcut  = abs(500000+f)- bandwidth/2 # 504000
highcut = abs(500000+f) + bandwidth/2 #  514000
sos_sum = signal.iirfilter(17, [lowcut, highcut], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

e_lowcut  = f-bandwidth/2
e_highcut = f+bandwidth/2
sos_efield = signal.iirfilter(17, [e_lowcut, e_highcut], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# w_e, h_e = signal.sosfreqz(sos, 2000, fs=Fs)                       
cut = 15
sos_dc = signal.iirfilter(17, [cut], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

ae_idx = abs(500000-f)
e_idx  = f

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

# rf_channel  = 0    
rf_channel  = 0
hydrophone_channel = 1 
i_channel   = 3
v_channel   = 4 
ae_channel  = 7  
e_channel   = 7 

resistor_current_mon = 49.9 
E_gap          = 0.002
AE_gap         = 0.0007
gain           = 1

filename = 'ptransient_stream_data.npy'
# filename = 'transient_stream_pdata2.npy'


d = np.load(filename)
a,b,c = d.shape
data = d.transpose(1,0,2).reshape(b,-1) 
p_d = data[hydrophone_channel]
rf_d = 10*data[rf_channel]

idx_h = np.argmax(p_d > 0.02)
idx_rf = np.argmax(rf_d > 20)

print ('idx:',idx_h,idx_rf)




filename = 'transient_stream_data.npy'
# filename = 'transient_stream_data_3s.npy'

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
ae_voltage_pp = np.pi*sum(np.fabs(ae_filtered))/len(ae_filtered)
AE_scale = (ae_voltage_pp/AE_gap)/gain
print ('AE V_x filtered pp (V)',ae_voltage_pp )
# 
# 
ae_sum_filtered = signal.sosfilt(sos_sum, data[ae_channel])
# 
# 
dc_filtered = signal.sosfilt(sos_dc, data[ae_channel])

fft_efield = fft(10*data[e_channel])
fft_e = np.abs(2.0/N * (fft_efield))[1:N//2]
# V_pp_fft = fft_e[e_idx]*2
# E_x = V_pp_fft/AE_gap
# print ('V_x(V) E_x(V/m): ',V_pp_fft,E_x)
e_filtered = signal.sosfilt(sos_efield, data[e_channel])
# V_x_scale = np.pi*sum(np.fabs(e_filtered))/len(e_filtered)
# E_x_scale = V_x_scale/AE_gap
# print ('V_x(V) scale E_x(V/m) scale:',V_x_scale,E_x_scale) 
print ('-----------AE Amplitude and E RMS -----------------')

aemax_pp_filtered = np.max(ae_filtered) - np.min(ae_filtered)
AE_max = (aemax_pp_filtered/AE_gap)/gain
print ('AE_filtered (V/m)',AE_max)

#  RMS E Amplitude Calculation
E_rms=np.sqrt(np.mean(e_filtered**2))/AE_gap  
print ('E rms(V/m)',E_rms)
print ('E/AE RMS ratio',E_rms/AE_max)


print ('------------PROBE V,I,R----------------')
V     = 10*data[v_channel]
I     = 5*data[i_channel]/resistor_current_mon
V_pp  = np.pi*sum(np.fabs(V))/len(V)
I_pp  = np.pi*sum(np.fabs(I))/len(I)
impedance = np.divide(V,I, where=I!=0)
impedance = np.abs(np.median(impedance))
print ('V_pp, I_pp(mA),R',V_pp, I_pp*1000,impedance)


idx_rf_final = np.argmax(10*data[rf_channel] > 20)
print ('idx_rf_final',idx_rf_final)


lag =  idx_rf_final - idx_rf 
t_offset = timestep * lag
print ('t_offset',t_offset)



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
(207, 207, 207),
(255, 255, 255),
(0, 0, 0) ]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableaucolorblind10)):    
    r, g, b = tableaucolorblind10[i]    
    tableaucolorblind10[i] = (r / 255., g / 255., b / 255.) 
# 
# 
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(2,1,1)
# ax.plot(t,data[ae_channel],'b')

# ax2 = fig.add_subplot(2,1,2)
# ax2.plot(t,dc_filtered,'b')
# plt.savefig("dc-removal.png", bbox_inches="tight") 
# plt.show()


fig, (ax2,ax3,ax4,ax5) = plt.subplots(nrows=4, sharex=True, subplot_kw=dict(frameon=False),figsize=(15,4.0)) # frameon=False removes frames
plt.subplots_adjust(hspace=.0)
# ax1.plot(t+t_offset,rf_d,color = tableaucolorblind10[0] )
ax2.plot(t+t_offset,p_d,color = tableaucolorblind10[3])
ax3.plot(t,data[v_channel],color = tableaucolorblind10[1])
ax4.plot(t,ae_sum_filtered*AE_gap/gain,color = tableaucolorblind10[4])
ax5.plot(t,ae_filtered*AE_gap/gain,color = tableaucolorblind10[5])
# ax6.plot(t,dc_filtered,color = tableaucolorblind10[6])

plt.xlabel('time(seconds)',fontsize=12)  

graph_names = ['RF Amplifier','Hydrophone','Applied Voltage','$\sum$ F','$\Delta$ F','DC filtered']  
startpt = -0.1    
# ax1.text(startpt, 0.0, graph_names[0], horizontalalignment='right',
#         verticalalignment='center',  fontsize=12, color=tableaucolorblind10[0]) 
# ax1.set_yticks([])         
ax2.text(startpt, 0.0, graph_names[1], horizontalalignment='right',
        verticalalignment='center',  fontsize=12, color=tableaucolorblind10[3])  
ax2.set_yticks([])
ax3.text(startpt, 0.0, graph_names[2], horizontalalignment='right',
        verticalalignment='center',  fontsize=12, color=tableaucolorblind10[1]) 
ax3.set_yticks([])
ax4.text(startpt, 0.0, graph_names[3], horizontalalignment='right',
        verticalalignment='center',  fontsize=12, color=tableaucolorblind10[4]) 
ax4.set_yticks([])
ax5.text(startpt, 0.0, graph_names[4], horizontalalignment='right',
        verticalalignment='center',  fontsize=12, color=tableaucolorblind10[5]) 
ax5.set_yticks([])
# ax6.text(startpt, 0.0, graph_names[5], horizontalalignment='right',
#         verticalalignment='center',  fontsize=12, color=tableaucolorblind10[6]) 
# ax6.set_yticks([])
plt.savefig("long-recording.svg", bbox_inches="tight",format="svg") 
plt.show()


