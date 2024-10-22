from drone_helper import connect_to_drone,GUIDED,moveToAlt,LocationGlobal

vehicle = connect_to_drone("udpout:10.42.0.1:10000")

vehicle.mode = GUIDED

print("Going")

# moveToAlt(vehicle,30.7477751,76.7568786,15)
loc = LocationGlobal(-30.7477751,76.7568786,15)

vehicle.simple_goto(loc)