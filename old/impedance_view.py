import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib
from scipy import interpolate
import serial, time
from subprocess import check_output
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq
import mouse_library as m

filename = 'impedances.npz'
d = np.load(filename,allow_pickle = True)


data = d['impedances']
frequencies =d['frequencies']
result = np.array(data)
print ('result: ',result.shape)

Is          = result[:,1]
phase       = result[:,5]
impedances  = result[:,6]

# ideally, I'd also like to see the amplitude of the measurement voltage. 
# currently, I think that is not submerged properly? At least I hope as I cannot see anything. 
# 
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(311)
plt.plot(frequencies,np.real(impedances),'b')
plt.plot(frequencies,np.imag(impedances),'r')
plt.legend(['resistance','reactance'],loc="upper left")
ax2 = fig.add_subplot(312)
plt.plot(frequencies,phase)
plt.legend(['phase'],loc="upper left")
ax3 = fig.add_subplot(313)
plt.plot(frequencies,Is)
plt.legend(['current'],loc="upper left")
plt.show()