'''

Title: field_view.py
Function: View a stream code data file. 

Author: Jean Rintoul
Date: 18.03.2022

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# 
# Fs = 1e5
# duration = 20.0
Fs = 4e6
duration =2.0

timestep = 1.0/Fs
N = int(Fs*duration)
# N = 30400000
print("expected no. samples:",N)
t = np.linspace(0, duration, N, endpoint=False)
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]

rf_channel  = 0    
v_channel   = 2
ae_channel  = 6



# filename = '8000.00_ati_stream_data.npy'
filename = '_stream.npy'


d = np.load(filename)
a,b,c = d.shape
data = d.transpose(1,0,2).reshape(b,-1) 
print (data.shape)


fft_efield = fft(data[v_channel])
fft_e = np.abs(2.0/N * (fft_efield))[1:N//2]

fft_rf = fft(data[rf_channel])
fft_rf = np.abs(2.0/N * (fft_rf))[1:N//2]

fft_ae = fft(data[ae_channel])
fft_ae = np.abs(2.0/N * (fft_ae))[1:N//2]

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
  
# Scale the RGB values to the [0, 1] ra                                                                                                                                                                                                                     nge, which is the format matplotlib accepts.    
for i in range(len(tableaucolorblind10)):    
    r, g, b = tableaucolorblind10[i]    
    tableaucolorblind10[i] = (r / 255., g / 255., b / 255.) 

# Change global font settings to help stylize the graph. 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2

# print (len(t), t[1000000],t[4000000])
# lag = 0.2 - 0.130194
# print ('lag: ',lag)

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(211)
plt.plot(frequencies, fft_ae, color = tableaucolorblind10[0],linewidth=2)
# plt.plot(frequencies, fft_e, color = tableaucolorblind10[1],linewidth=2)
plt.xlim([0,1000000])
# plt.xlim([0,100])
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2 = fig.add_subplot(212)
plt.plot(t, data[rf_channel], color = tableaucolorblind10[0],linewidth=2)
plt.plot(t, data[ae_channel], color = tableaucolorblind10[0],linewidth=2)
# plt.plot(t, data[v_channel], color = tableaucolorblind10[1],linewidth=2)
# plt.axvline(x=0.2)
# plt.axvline(x=0.8)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
save_title = '_signals'
plt.savefig(save_title+".svg", format="svg") 
plt.savefig(save_title+".png")
plt.show()

