from dronekit import connect,VehicleMode,LocationGlobalRelative
import time

MAV_CMD_NAV_RETURN_TO_LAUNCH = 20

vehicle = connect('tcp:localhost:5763')
vehicle.wait_ready(True,timeout=300)

while not vehicle.is_armable:
    print("Waiting")
    time.sleep(1)

# vehicle.mode = VehicleMode("GUIDED")
# vehicle.armed = True

# while not vehicle.armed:
#     print("Arming")
#     time.sleep(1)

# vehicle.simple_takeoff(30)
# time.sleep(15)

vehicle.commands.download()
vehicle.commands.wait_ready()

vehicle.commands.next = 0
vehicle.mode = VehicleMode('AUTO')

print('Starting Mission')
interupted = False
while(True):
    nextwaypoint = vehicle.commands.next
    command = vehicle.commands[nextwaypoint-1].command
    print(f"{nextwaypoint}: {command}")
    if(nextwaypoint == 5 and not interupted):
        print("Interupting flight path")
        current_location = vehicle.location.global_frame

        vehicle.mode = VehicleMode('GUIDED')
        vehicle.wait_for_mode(VehicleMode('GUIDED'))
        point = LocationGlobalRelative(13.3946242,77.7318871,30)
        vehicle.simple_goto(point)
        time.sleep(7)
        vehicle.simple_goto(current_location)
        time.sleep(7)
        vehicle.mode = VehicleMode('AUTO')
        print("Continuing Mission")
        interupted = True
        vehicle.commands.next = 5

    if(command==MAV_CMD_NAV_RETURN_TO_LAUNCH):
        break
    time.sleep(1)

print('Mission End')
vehicle.mode = VehicleMode('RTL')

vehicle.close()