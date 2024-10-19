from drone_helper import *

vehicle = connect_to_drone("tcp:10.42.0.1:5760")

print("Get some vehicle attribute values:")
print(" GPS: %s" % vehicle.gps_0)
print(" Battery: %s" % vehicle.battery)
print(" Last Heartbeat: %s" % vehicle.last_heartbeat)
print(" Is Armable?: %s" % vehicle.is_armable)
print(" System status: %s" % vehicle.system_status.state)
print(" Mode: %s" % vehicle.mode.name)    # settable

print(vehicle.channels)

for i in range(1,1000):
    vehicle.channels.overrides = { '8' : 1100 }
# vehicle.channels.overrides['8'] = 1100
# print(vehicle.channels.overrides)
# vehicle.channels.overrides['15'] = 1100
time.sleep(1)

# Close vehicle object before exiting script
vehicle.close()