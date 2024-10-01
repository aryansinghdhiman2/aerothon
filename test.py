from dronekit import connect,VehicleMode,LocationGlobalRelative
import time
vehicle = connect('tcp:localhost:5763')
vehicle.wait_ready(True,timeout=300)

while not vehicle.is_armable:
    print("Waiting")
    time.sleep(1)



vehicle.mode = VehicleMode("GUIDED")
vehicle.armed = True
print(vehicle.mode.name)

while not vehicle.armed:
    print("Arming")
    time.sleep(1)

vehicle.simple_takeoff(30)
time.sleep(15)
# vehicle.mode = VehicleMode('LAND')
point = LocationGlobalRelative(13.3946242,77.7318871,30)
vehicle.simple_goto(point)

# vehicle.mode = VehicleMode("AUTO")
# time.sleep(30)

# point = LocationGlobalRelative(30.7461705,76.7564592,100)
# vehicle.simple_goto(point)
# time.sleep(30)

time.sleep(60)

vehicle.mode = VehicleMode("RTL")
vehicle.close()