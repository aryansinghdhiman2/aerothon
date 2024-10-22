from drone_helper import connect_to_drone,moveToAlt,GUIDED,LocationGlobal
import time
from dronekit import LocationGlobalRelative

# vehicle = connect_to_drone("tcp:localhost:5762")
vehicle = connect_to_drone("udpout:10.42.0.1:10000")

# vehicle.message_factory.system_time_send(time.time(),time.time())


print("Going")
vehicle.mode = GUIDED

# l = LocationGlobalRelative(-30.7479417,76.7565145,20)

moveToAlt(vehicle,30.7477394,76.7567633,15)
# vehicle.simple_goto(l)
# print(vehicle.location.global_frame)
# print(vehicle.location.local_frame)
# print(vehicle.location.global_relative_frame)
