import time
import pigpio

IN_GPIO=4
OUT_GPIO=[5, 6, 7, 8, 9, 10, 11, 12]

start_of_frame = False
channel = 0
last_tick = None

def cbf(gpio, level, tick):
   global start_of_frame, channel, last_tick
   if last_tick is not None:
      diff = pigpio.tickDiff(last_tick, tick)
      if diff > 3000: # start of frame
         start_of_frame = True
         channel = 0
      else:
         if start_of_frame:
            if channel < len(OUT_GPIO):
               pi.set_servo_pulsewidth(OUT_GPIO[channel], diff)
               print(channel, diff)
               channel += 1
   last_tick = tick

pi = pigpio.pi()
if not pi.connected:
   exit()

pi.set_mode(IN_GPIO, pigpio.INPUT)

cb = pi.callback(IN_GPIO, pigpio.RISING_EDGE, cbf)

time.sleep(60)

cb.cancel()

pi.stop()
