#!/usr/bin/python
'''
 Author: Jean Rintoul Date: 24/04/2021

 Ramp through a range of voltage amplitudes, collecting 10 measurements in each category, to estimate how the applied voltage effects
 the demodulation. 
 

'''
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib
from scipy import interpolate
import acoustoelectric_library as m
import serial, time
from subprocess import check_output
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq

def find_nearest(array, value):
    idx = min(range(len(array)), key=lambda i: abs(array[i]-value))
    return idx

def get_AE_phi(fileprefix,duration,Fs,current_frequency):
  N             = Fs*duration
  ae_channel    = 6  
  e_channel     = 6 
  transducer_frequency = 500000 
  f             = current_frequency  # this will be a bit innaccurate for demodulation as there are 3 frequencies - I will need to post process the files. 
  # 
  timestep = 1.0/Fs
  N = int(N)
  xf = np.fft.fftfreq(N, d=timestep)[:N//2]
  frequencies   = xf[1:N//2]
  ae_diff_idx   = find_nearest(frequencies,transducer_frequency-f)
  ae_sum_idx    = find_nearest(frequencies,transducer_frequency+f)
  e_idx         = find_nearest(frequencies,f)
  # 
  # Load the file. 
  filename = fileprefix+'_stream.npy'
  print ('filename is',filename)
  d = np.load(filename,allow_pickle = True)
  a,b,c = d.shape
  data = d.transpose(1,0,2).reshape(b,-1) 
  print ('data shape',data.shape)  
  
  fft_efield = fft(data[e_channel])
  fft_e = np.abs(2.0/N * (fft_efield))[1:N//2]
  fft_aefield = fft(data[ae_channel])
  fft_ae = np.abs(2.0/N * (fft_aefield))[1:N//2]

  AE_sum_volts  = 2*fft_ae[ae_sum_idx]
  AE_diff_volts = 2*fft_ae[ae_diff_idx]
  E_volts       = 2*fft_e[e_idx]

  return E_volts,AE_diff_volts,AE_sum_volts

def make_ati_demod_evsae_recording(repeat_no,current_frequency,duration,Fs,fileprefix = ''):

  fileprefix = str(round(current_frequency,2))+'_f_'+str(repeat_no)
  str1 = 'marker_stream -p '  
  command = str1+fileprefix
  str2 = ' -f '
  command = command + str2 + str(current_frequency)
  print ('command check:',command)

  result = None
  try:
     foo = check_output(command, shell=True)
     result = str(foo, 'utf-8')
     # print (result)
  except Exception as e: print(e)

  # so the name will be 0.1_10_stream.npz (volts followed by repeat number )
  e_field = 0  
  ae_diff = 0 
  ae_sum  = 0 
  try:
     e_field,ae_diff,ae_sum = get_AE_phi(fileprefix,duration,Fs,current_frequency)
  except Exception as e: print(e)

  return result,e_field,ae_diff,ae_sum

#  Define some constants
Fs          = 5e6
duration    = 0.1
N           = Fs*duration
ae_channel  = 6
e_channel   = 6 

no_repeats  = 4


spectrum_range = np.round(np.linspace(10000,490000,num=20)).astype(int)
# spectrum_range = np.round(np.linspace(280000,500000,num=20)).astype(int)
print ('spectrum range:',spectrum_range)


p_z_start = 181.1
p_y_start = -19.8 # remember y is the z dimension. 
p_x_start = 5.0

# orthogonal electrode set up. 
offset_x    = 1.5
offset_z    = -49.0
offset_y    = -4.5



m.turn_printer_on()
time.sleep(5) 
m.move(0.0,axis='y')
m.deltahome(axis='x')

m.move(p_z_start+offset_z,'z')
m.move(p_x_start+offset_x,'x')
m.move(p_y_start+offset_y,'y')

time.sleep(5) 
m.turn_printer_off()


data = []
for i in range(len(spectrum_range)): 
  current_frequency = round(spectrum_range[i],0)
  print('current frequency: ',current_frequency)  
  # Do 10 repeats at each voltage setting so that I can do more decent stats.  
  for j in range(no_repeats):
    repeat_no     = int(j) 
    result        = None
    while result is None:
      try:
        result,E_volts,AE_diff_volts,AE_sum_volts = make_ati_demod_evsae_recording(repeat_no,current_frequency,duration,Fs)
        print ('efield/ae_result:',current_frequency,E_volts,AE_diff_volts,AE_sum_volts)
        if E_volts != 0:
          d = [current_frequency,E_volts,AE_diff_volts,AE_sum_volts]
          data.append(d)
      except:
        pass

print ('all recordings made. Done!')
# 
# print (d)
# print ('data:', data)
# 
outfile="frequency_ae.npz"  
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
np.savez(outfile,data = data)
print ('final data is: ')
data_array = np.array(data)
print (data_array.shape)
f   = data_array[:,0]
e   = data_array[:,1]
ae  = data_array[:,2]
print (len(f),ae)
# 
# 
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
plt.plot(f,e,'b')
ax.set_ylabel('(V/m)')
ax2 = fig.add_subplot(2,1,2)
plt.plot(f,ae,'r')
ax2.set_ylabel('(V/m)')
ax2.set_xlabel('current amplitude(V)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.show()
