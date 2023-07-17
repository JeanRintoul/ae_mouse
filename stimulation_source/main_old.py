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

# Select the frequency it will run at. 
frequency            = 1
led_start_time       = 10000   # in milliseconds(ms)
led_stop_time        = 80000 

# Initialize the various inputs and outputs.
#global led,stimulus_monitor,start_time,check,skip_one
check                = 0
start_time           = 0
skip_one             = 0
pin_event            = 28
pin_led              = 14
pin_stimulus_monitor = 8
pin_button           = 29
tim              = Timer()
led              = Pin(pin_led,Pin.OUT)
stimulus_monitor = Pin(pin_stimulus_monitor,Pin.OUT)
button           = Pin(pin_button, machine.Pin.IN, machine.Pin.PULL_UP)
event            = Pin(pin_event, machine.Pin.IN, machine.Pin.PULL_UP)
led.value(0)



def tick(timer):
    global led,stimulus_monitor,start_time,check,skip_one
    time_elapsed = time.ticks_ms()-start_time
    if led_start_time < time_elapsed < led_stop_time:
        if skip_one % 4 == 0: # turn on every third one. 
           led.value(1)
           stimulus_monitor.value(1)
        else:
           led.value(0)
           stimulus_monitor.value(0)
        skip_one = skip_one + 1   
    else:
        led.value(0)
        stimulus_monitor.value(0)
    if check == 0:
        print ('time elapsed: ', time_elapsed)
        check = 1
    
def button_tick(timer):
    global led,stimulus_monitor,start_time,check,skip_one
    if skip_one % 4 == 0: # turn on every third one. 
        led.value(1)
        stimulus_monitor.value(1)
    else:
        led.value(0)
        stimulus_monitor.value(0)
    skip_one = skip_one + 1   
        
def all_off(timer):
    led.value(0)
    stimulus_monitor.value(0)
    
@rp2.asm_pio()
def wait_pin_low():
    wrap_target()
    wait(0, pin, 0)
    irq(block, rel(0))
    wait(1, pin, 0)
    wrap()

@rp2.asm_pio()
def wait_pin_high():
    wrap_target()
    wait(1, pin, 0)
    irq(block, rel(0))
    wait(0, pin, 0)
    wrap()
# 
def pin_low_handler(sm):
    print ('pin low handler called')
    global start_time,check
    tim.init(freq=2*frequency,mode=Timer.PERIODIC,callback=tick)
    start_time = time.ticks_ms()
    print(time.ticks_ms(), sm)
#     
def pin_high_handler(sm):
    global check
    print ('pin high handler called')
    if check == 1:
        led.value(0)
        stimulus_monitor.value(0)
        tim.init(callback=None)
        check = 0

         
def button_handler(sm):
    global start_time,check
    # print ('button handler called',check)
    if check == 1:
        led.value(0)
        stimulus_monitor.value(0)
        tim.init(callback=None)
        check = 0
        return
    if check == 0:
        tim.init(freq=2*frequency,mode=Timer.PERIODIC,callback=button_tick)
        start_time = time.ticks_ms()
        print(time.ticks_ms(), sm)
        check = 1

#   starts and stops of the daq. 
sm0 = rp2.StateMachine(0, wait_pin_high, in_base=event)
sm0.irq(pin_high_handler)
sm1 = rp2.StateMachine(1, wait_pin_low, in_base=event)
sm1.irq(pin_low_handler)

# Instantiate StateMachine on Pin(button) toi take care of button press events. 
sm2 = rp2.StateMachine(2, wait_pin_low, in_base=button)
sm2.irq(button_handler)
# 
time.sleep(1)
# Start the StateMachine's running.
sm0.active(1)
sm1.active(1)
sm2.active(1)


    
    

