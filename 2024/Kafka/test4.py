from drone_helper import *
from math import degrees
from pymavlink.dialects.v10.common import MAV_FRAME_BODY_FRD,MAV_CMD_DO_REPOSITION,MAV_FRAME_LOCAL_NED,MAV_CMD_NAV_WAYPOINT,MAV_FRAME_GLOBAL_RELATIVE_ALT,MAV_FRAME_LOCAL_FRD,MAV_FRAME_BODY_NED,MAV_FRAME_BODY_OFFSET_NED
from coordinate_conversion import calculate_body_ned_coordinates


vehicle = connect_to_drone("udpout:10.42.0.1:10000")

@vehicle.on_message("RC_CHANNELS")
def channel_handler(vehicle:Vehicle,name,message):
    print("Name ",name)
    print("Message: ",message.chan6_raw)
    # message.chan6_raw

print("Get some vehicle attribute values:")
print(" GPS: %s" % vehicle.gps_0)
print(" Battery: %s" % vehicle.battery)
print(" Last Heartbeat: %s" % vehicle.last_heartbeat)
print(" Is Armable?: %s" % vehicle.is_armable)
print(" System status: %s" % vehicle.system_status.state)
print(" Mode: %s" % vehicle.mode.name)    # settable



time.sleep(1)

while(True):
    time.sleep(0.5)
# Close vehicle object before exiting script
vehicle.close()