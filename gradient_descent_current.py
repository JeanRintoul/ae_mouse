"""
    
    Voltage source based current calibration: 
    Online stochastic gradient descent: Given a starting voltage and a desired current, 
    find the voltage at which the desired current occurs. 

    Note: this only works on pure sine waves as it uses the FFT at the specified frequency to determine amplitude. 

    By Jean Rintoul
    Date: April 13 2021

"""
import matplotlib.pyplot as plt
import numpy as np
from subprocess import check_output
import mouse_library as m
from scipy.fft import fft, fftfreq

# remember to update these: 
test_no = 1
current_frequency = 2000 

aeti_variables = {
'type':'impedance',         # choice of 'pressure' or 'ae'. Note: this doesn't change the contents of the file, just the way it is processed at the end. 
'Fs': 1e5,                  # 
'duration': 1.0,            # 
'position': test_no,
'pressure_amplitude': 0.0,
'pressure_frequency': 10000.0,
'current_amplitude': 0.1,   #  its actually a voltage .. Volts. 0.0002
'current_frequency': current_frequency,
'ti_frequency': 0,       # if this is included or  > 0 it means we are adding two sine waves together. i.e. TI. 
'ae_channel': 0,            # the channel of the measurement probe. 
'e_channel': 4,             # this is the voltage measured between the stimulator probes. 
'rf_monitor_channel': 1,    # this output of the rf amplifier. 
'current_monitor_channel': 5,  # this is the current measurement channel of the transformer. 
'start_pause': 0.1,         # percent of file in ramp mode. 
'end_pause': 0.9,           # 
'no_ramp':1.0,              # when we have no ramp we set this to 1. i.e. an impedance spectrum test. 
'gain':1,                   # this is the preamp gain. If not using a preamp, set it to 1.
'IV_attenuation':10,        # the current and voltage monitor both have attenuators on them 
'marker_channel':2,
'command_c':'mouse_stream',
'save_folder_path':'D:\\mouse_aeti\\e74_nr_ketxyl_screw_mouse',
'experiment_configuration':'monopolar',  # if it is monopolar, it is coming straight from the fg, bipolar, goes through David Bono's current source. 
}

# 
# Variables: 
# 
desired_current = 0.03 # peak to peak milliamperes. 
print ('Desired current is:',desired_current,'(mA)')
cur_v           = 0.1 # 0.5 # The algorithm starts at this voltage 
rate            = 0.05 # Learning rate
precision       = 0.001 # This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters       = 15 # maximum number of iterations
iters           = 0 # iteration counter

Fs       = aeti_variables['Fs'] 
duration = aeti_variables['duration'] 
i_channel = aeti_variables['current_monitor_channel'] 
marker_channel = aeti_variables['marker_channel']
m_channel = aeti_variables['ae_channel'] 
v_channel = aeti_variables['e_channel'] 
savepath = aeti_variables['save_folder_path']
attenuation = aeti_variables['IV_attenuation']
# 
# Core gradient descent code. 
i = 0 
def measurement(v): # 
    global i,data,Z
    v = round(abs(v),2)
    aeti_variables['current_amplitude'] = v 
    result, data_out  = m.aeti_recording(**aeti_variables)
    # print ('impedance:',data_out[0])
    data,idx_lag,original_data  = m.align_data_to_marker_channel(**aeti_variables)
    [V_pp,I_pp,Z,L,C,phase_shift,resistance,reactance,measurement_pp,phase_shift2]=data_out[0]
    i = 1000*I_pp  # Convert to mA. 
    return i 
# 
# this is the function we are minimizing: the difference between the desired current and the measured current.  
df = lambda v: measurement(v) - desired_current 
# Currently it finds the minima the measured current at a particular voltage - the desired voltage.  
# 
voltage_history = []
cost_history = []
while previous_step_size > precision and iters < max_iters:
    prev_v = cur_v # Store current v value in prev_v
    if prev_v >= 12.0: 
        prev_v = 12  # This is the maximum of the tiepie output 
        print ('reached maximum voltage')
    print ('stuff',cur_v, rate, prev_v)
    cur_v = cur_v - rate * df(prev_v) # Gradient descent
    if cur_v < 0: 
        cur_v = 0 # Since we have a symmetrical wave, the value doesn't go below zero.  
    previous_step_size = abs(cur_v - prev_v) # Change in v
    # print ('cur_v prev_v',cur_v,prev_v)
    # print ('i',i, desired_current)
    # variable change rate so that it converges more quickly. 
    print ('change rate',desired_current/i)
    change_rate = desired_current/i
    if change_rate < 0.5 or change_rate > 2:  
        print ('large rate')
        rate = 0.7
    else: 
        rate = 0.1 
    # 
    cost_history.append(previous_step_size)
    voltage_history.append(cur_v)
    iters = iters+1 #iteration count
    print("Iteration:",iters," Current=",i,"(mA) @ V=",cur_v,"(V)","Z (Ohms) = ",round(Z,2)) #Print iterations
    
print("Converged at", round(cur_v,4)," Volts")
print("At", round(cur_v,4),"Volts current is", round(i,4),"milliamperes","Impedance is",round(Z,2) )


resistor_current_mon    = 49.9  # 49.9 Ohms for current monitor, 
i_data         = -5*data[i_channel]/resistor_current_mon
i_data         = 1000*i_data # convert to mA. 
v_data         = -10*data[v_channel]
if aeti_variables['experiment_configuration'] == 'monopolar':
   print ('monopolar mode')
   resistor_current_mon    = 47  # 49.9 Ohms for current monitor, 
   i_data                  = -data[i_channel]/resistor_current_mon
   i_data                  = 1000*i_data # convert to mA. 
   v_data                  = -attenuation*data[v_channel]



timestep = 1.0/Fs
N = int(Fs*duration)
t = np.linspace(0, duration, int(N), endpoint=False)

xf = np.fft.fftfreq( (N), d=timestep)[:(N)//2]
frequencies = xf[1:(N)//2]
fft_i = fft(i_data)
fft_i = np.abs(2.0/(N) * (fft_i))[1:(N)//2]

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(311)
ax.plot(voltage_history,'b', marker='.' )
axtwo = ax.twinx()
axtwo.plot(cost_history,'r', marker='.' )
plt.xlabel('iteration')
# plt.ylabel('difference from desired value')
ax.set_ylabel('Applied Volts History(V)',color='b')
axtwo.set_ylabel('Cost history(V)',color='r')

ax2 = fig.add_subplot(312)
plt.plot(t,v_data,'b')
plt.plot(t,i_data,'r')
ax2.legend(['V(V)','I(ma)'],loc='upper right')
plt.xlabel('time(s)')
plt.suptitle('Cost function')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax3 = fig.add_subplot(313)
plt.plot(frequencies, fft_i,color='b')
ax3.set_xlim([0,4000])
ax3.set_ylabel('current (mA)')

plot_filename = savepath + '\\t'+str(test_no)+'_gradient_descent.png'
plt.savefig(plot_filename,bbox_inches='tight')
plt.show()
