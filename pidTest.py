from drone_helper import *
from dronekit import LocationGlobal
from time import sleep
veh = connect_to_drone(def_address)
controller = configure_pid(20000,17400,3,3)

arm(veh)
veh.simple_takeoff(10)

print("Took off")
sleep(3)


target = LocationGlobal(30.7478496,76.7565248)

while(True):
    current_location = getCurrentLocation(veh)
    delta_x = current_location.lat - target.lat
    delta_y = current_location.lon - target.lon
    if(delta_x < 0.000001 and delta_y < 0.000001):
        break

    roll = controller.roll(delta_x)
    pitch = controller.pitch(delta_y)

    print(f"LAT: {current_location.lat}, LON: {current_location.lon}, DX: {delta_x}, DY: {delta_y}, ROLL: {roll}, PITCH: {pitch}")

    sendRollAndPitch(veh,roll,pitch)
    sleep(1)

