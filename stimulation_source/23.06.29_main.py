# Stimulus onset system
#  Date:   01/08/2022
#  Author: Jean Rintoul 
#
#   - PIO wrapping
#   - PIO wait instruction, waiting on an input pin
#   - PIO irq instruction, in blocking mode with relative IRQ number
#   - setting the in_base pin for a StateMachine
#   - setting an irq handler for a StateMachine
#   - instantiating 2x StateMachine's with the same program and different pins
from machine import Pin
from machine import Timer
import rp2
import time
# 
# Marker to show when US is turned on and off. Update from my main code to suit specific circumstance
duration   = 8.0 
start_null = 0.125 * duration * 1000 # in milliseconds(ms)
end_null   = 0.875 * duration * 1000 # in milliseconds(ms)
# 
# Select the frequency it will run at. 
# frequency            = 1      # 0.5s on duration. 1 on every 2 seconds.
frequency            = 2        # 0.25s duration, 1 on every second.
# frequency            = 4        # 0.125 duration, 1 on every 0.5s. 
# frequency            = 8        # 0.0625 duration, 1 on every 0.25s.
# frequency            = 7       # 0.0625 duration, 1 on every 0.25s.
#frequency              = 8      # 0.0625 duration, 1 on every 0.25s.
# frequency            = 16       # 0.03125 duration, 1 on every 0.125s. 
# frequency            = 32       # 0.015625 duration, 1 on every 0.0625s. 
# frequency            = 64       # 0.0078125 duration, 1 on every 0.03125s. 
# led_start_time       = 500   # in milliseconds(ms)
led_start_time       = 400   # in milliseconds(ms)
led_stop_time        = 10000
# led_stop_time        = 4000
# led_start_time       = 5000   # in milliseconds(ms)
# led_stop_time        = 55000 
stim_monitor_marker_frequency = 1000 # in milliseconds(ms)

# Initialize the various inputs and outputs.
#global led,stimulus_monitor,start_time,check,skip_one
check                = 0
start_time           = 0
skip_one             = 0
pin_event            = 28
pin_led              = 14
pin_stimulus_monitor = 8
pin_button           = 29
pin_on_led           = 15 # this is the led that. 
tim              = Timer()
led              = Pin(pin_led,Pin.OUT)
led_on           = Pin(pin_on_led,Pin.OUT)
stimulus_monitor = Pin(pin_stimulus_monitor,Pin.OUT)
button           = Pin(pin_button, machine.Pin.IN, machine.Pin.PULL_UP)
event            = Pin(pin_event, machine.Pin.IN, machine.Pin.PULL_UP)
led.value(0)
led_on.value(0)


def tick(timer):
    global led,stimulus_monitor,start_time,check,skip_one
    time_elapsed = time.ticks_ms()-start_time
    if led_start_time < time_elapsed < led_stop_time:
        # if skip_one % 4 == 0: # turn on one in 4.
        if skip_one % 2 == 0:   # turn on one in 2 
           led.value(1)
           stimulus_monitor.value(1)
        else:
           led.value(0)
           stimulus_monitor.value(0)
        skip_one = skip_one + 1   
    else:
        led.value(0)
        stimulus_monitor.value(0)
    #     
    # this part is to mark when the ultrasound is turned on or not.    
    if time_elapsed > start_null and time_elapsed < end_null:
        led_on.value(1)
    else:
        led_on.value(0)
        
    #         
    if check == 0:
        print ('time elapsed: ', time_elapsed)
        check = 1

# 
def button_tick(timer):
    global led,stimulus_monitor,start_time,check,skip_one
    if skip_one % 2 == 0: # turn on every third one. 
        led.value(1)
        stimulus_monitor.value(1)
    else:
        led.value(0)
        stimulus_monitor.value(0)
    skip_one = skip_one + 1   
#  
def all_off(timer):
    led.value(0)
    led_on.value(0)    
    stimulus_monitor.value(0)
# 
@rp2.asm_pio()
def wait_pin_low():
    wrap_target()
    wait(0, pin, 0)
    irq(block, rel(0))
    wait(1, pin, 0)
    wrap()
# 
@rp2.asm_pio()
def wait_pin_high():
    wrap_target()
    wait(1, pin, 0)
    irq(block, rel(0))
    wait(0, pin, 0)
    wrap()
# this happens when the daq starts. 
def pin_low_handler(sm):
    print ('pin low handler called, daq event')
    global start_time,check
    # marker for start time. 
    stimulus_monitor.value(1)
    led_on.value(1)
    time.sleep(0.01)
    stimulus_monitor.value(0)
    led_on.value(0)    
    # 
    tim.init(freq=frequency,mode=Timer.PERIODIC,callback=tick)
    start_time = time.ticks_ms()
    # led_on.value(1)
    print(time.ticks_ms(), sm)
#     
def pin_high_handler(sm):
    global check
    print ('pin high handler called, ready to receive event')
    
    # this is the final LED flash when the end of daq recording is reached. 
    led_on.value(1)
    time.sleep(0.01)
    led_on.value(0)
        
    if check == 1:
        led.value(0)
        led_on.value(0)
        stimulus_monitor.value(0)
        tim.init(callback=None)
        check = 0

#     
def button_handler(sm):
    global start_time,check
    # print ('button handler called',check)
    if check == 1:
        led.value(0)
        led_on.value(0)
        stimulus_monitor.value(0)
        tim.init(callback=None)
        check = 0
        return
    if check == 0:
        tim.init(freq=frequency,mode=Timer.PERIODIC,callback=button_tick)
        start_time = time.ticks_ms()
        print(time.ticks_ms(), sm)
        led_on.value(1)
        check = 1

# starts and stops of the daq. 
sm0 = rp2.StateMachine(0, wait_pin_high, in_base=event)
sm0.irq(pin_high_handler)
#  
sm1 = rp2.StateMachine(1, wait_pin_low, in_base=event)
sm1.irq(pin_low_handler)

# Instantiate StateMachine on Pin(button) toi take care of button press events. 
sm2 = rp2.StateMachine(2, wait_pin_low, in_base=button)
sm2.irq(button_handler)

# 

# wait for a second. 
time.sleep(1)

# Start the StateMachine's running.
sm0.active(1)
sm1.active(1)
sm2.active(1)


    
    

