"""
    
    Voltage source based current calibration: 
    Online stochastic gradient descent: Given a starting voltage and a desired current, 
    find the voltage at which the desired current occurs. 

    By Jean Rintoul
    Date: April 13 2021

"""
import matplotlib.pyplot as plt
import numpy as np
from subprocess import check_output

desired_current = 2.5 # peak to peak milliamperes. 
print ('Desired current is:',desired_current,'(mA)')


cur_v = 0.5 # The algorithm starts at voltage = 1V
rate = 0.1 # Learning rate
precision = 0.01 # This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters = 10 # maximum number of iterations
iters = 0 #iteration counter

# Data check code. 
# data = np.load('gdescent_block.npy')
# a,b = data.shape
# print (data.shape)
# resistor_current_mon = 49.9 
# ma_e1 = 1000*5*data[3]/resistor_current_mon
# I_pp = max(ma_e1)-min(ma_e1)
# print ('I est starting',I_pp)
# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111)
# plt.plot(ma_e1,'r')
# plt.show()

# def measurement(v): # 
#     return v**2
i = 0 
# this will be found through experiment. i.e. running the c log code. 
def measurement(v): # 
    v = round(abs(v),2)
    str1 = 'gradient_descent -a '
    command = str1+str(v)
    # print ('command:',command)
    foo = check_output(command, shell=True)
    result = str(foo, 'utf-8')
    # print (result)
    ind = result.find('(mA)')
    current = float(result[ind+5:].strip())
    # print ('i: ',current)
    global i 
    i = current
    return i 

# this is the function we are minimizing. 
# it's the difference between the desired current and the measured current.  
df = lambda v: measurement(v) - desired_current 
# Currently it finds the minima the measured current at a particular voltage - the desired voltage.  

#  So if i - 8ma. df = 8-2 = 6 
#  if rate = 0.2      cur_v = 0.5
# 
# 0.5 - 0.2(6)  = 0.5 - 3  = -2.5
# if it's negative, set it to zero... 

cost_history = []
while previous_step_size > precision and iters < max_iters:
    prev_v = cur_v #Store current v value in prev_v
    if prev_v >= 12.0: 
        prev_v = 12  # This is the maximum of the tiepie output 
        print ('reached maximum voltage')
    cur_v = cur_v - rate * df(prev_v) #Grad descent
    if cur_v < 0: 
        cur_v = 0 # Since we have a symmetrical wave, the value doesn't go below zero.  
    previous_step_size = abs(cur_v - prev_v) #Change in v
    cost_history.append(previous_step_size)
    iters = iters+1 #iteration count
    print("Iteration:",iters," Current=",i,"(mA) @ V=",cur_v,"(V)") #Print iterations
    
print("Converged at", round(cur_v,4)," Volts")
print("At", round(cur_v,4),"Volts current is", round(i,4),"milliamperes")

# TODO: to get greater accuracy I should do a find peaks funciton in python to average between peaks. 

data = np.load('gdescent_block.npy')
a,b = data.shape
# print (data.shape)
resistor_current_mon = 49.9 

ma_e1 = 5*data[3]/resistor_current_mon
V = 10*data[5]

V_pp = np.max(V)-np.min(V)
I_pp = (np.max(ma_e1 )-np.min(ma_e1))
Z = np.abs(V_pp /I_pp)
print ('max I(mA),V(V),Z(Ohms)', 1000*I_pp,V_pp,Z)


# impedance = np.divide(V,ma_e1, where=ma_e1!=0)
# print ('Load Impedance is (Ohms): ', np.abs(np.median(impedance)))
impedance = Z  
print ('Impedance is:',Z)
Fs = 1e8
duration = 0.008
N = Fs*duration
t = np.linspace(0, duration, int(N), endpoint=False)


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(211)
plt.plot(cost_history,'b', marker='.' )
plt.xlabel('iteration')
plt.ylabel('difference from desired value')
ax2 = fig.add_subplot(212)
plt.plot(t,1000*ma_e1,'r')
plt.plot(t,V,'b')
ax2.legend(['I(ma)','V(V)'],loc='upper right')
plt.xlabel('time(s)')
plt.suptitle('Cost function')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plot_filename = 'gradient_descent.png'
plt.savefig(plot_filename)
plt.show()
