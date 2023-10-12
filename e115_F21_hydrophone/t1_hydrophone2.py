import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq,fftshift,ifft,ifftshift
from scipy.signal import blackman
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.signal import fftconvolve
from sklearn.linear_model import LinearRegression
#
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 16

# Data from t1. F21 attenuation plots. 
noF21           = [369,399,396,457]
thicknesses     = [0,1.3,7,11.3] # mm
thin            = [295,313,300,380]
# thin_control    = [369,399,396,457 ]
thin_control = noF21
medium          = [270, 294, 277, 307]
# medium_control  = [427, 423, 412, 421]
medium_control  = thin_control
thick           = [237, 243, 245, 231]
# thick_control   = [338, 350, 360, 346]
thick_control = thin_control


thin_attenuation   = np.array(thin)/np.array(thin_control)
medium_attenuation = np.array(medium)/np.array(medium_control)
thick_attenuation  = np.array(thick)/np.array(thick_control)
nothing_attenuation = np.array(noF21)/np.mean(noF21)
# np.array([1,1,1,1])

F21 = np.array([noF21,thin,medium,thick])
F21_attenuation = np.array([nothing_attenuation,thin_attenuation,medium_attenuation,thick_attenuation])
print (F21)
F21_mean = np.mean(F21,1)
F21_attenuation_mean = 1 - np.median(F21_attenuation,1)
F21_std = np.mean(F21,1)

x = np.array(thicknesses)
y = F21_mean
# find line of best fit
a, b = np.polyfit(x, y, 1)
print ('pressure line:gradient and add',a,b)


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(thicknesses,F21,'.k')
plt.plot(x, a*x+b,'k')  
# ax.set_ylabel('Pressure(Pa)')
# ax.set_xlabel('Thickness(mm)')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.tight_layout()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("F21_pressure_vs_thickness.png", bbox_inches="tight")
plt.show()


x = np.array(thicknesses)
y = 100*F21_attenuation_mean
#find line of best fit
a, b = np.polyfit(x, y, 1)
print ('attenuation line:gradient and add',a,b)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(thicknesses,100*(1 - F21_attenuation),'.k')
plt.plot(x, a*x+b,'k')  
# ax.set_ylabel('Pressure(Pa)')
# ax.set_xlabel('Thickness(mm)')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.set_ylim([0,1.1*100*np.max(1 - F21_attenuation)])
plt.tight_layout()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("F21_attenuation_vs_thickness.png", bbox_inches="tight")
plt.show()


