from drone_helper import *
import pigpio
import time
vehicle = connect_to_drone_serial("/dev/ttyACM0",57600)
pi = pigpio.pi()

@vehicle.on_message('RC_CHANNELS')
def chin_listener(self, name, message):
 # print '%s attribute is: %s' % (name, message)
 pitch = 0
 roll = 0
 mod = 0
 flag_rec = 0

 
 pitch=message.chan2_raw    # copter (throttle)
 print(f'channels:{pitch}')
 if(pitch > 1700):
     pi.set_servo_pulsewidth(12,2500)
 if(pitch < 1400):
     pi.set_servo_pulsewidth(12,500)
     

while(True):
    #vehicle.mode = GUIDED
    #vehicle.wait_for_mode(GUIDED)
    #print(vehicle.channels)
    #time.sleep(1)
    #vehicle.mode = RTL
    #vehicle.wait_for_mode(RTL)
    time.sleep(1)
    
vehicle.close()
pi.set_servo_pulsewidth(12,0)
pi.stop()