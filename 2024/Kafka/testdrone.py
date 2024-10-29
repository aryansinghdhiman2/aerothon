from drone_helper import *
from math import degrees
from pymavlink.dialects.v10.common import MAV_FRAME_BODY_FRD,MAV_CMD_DO_REPOSITION,MAV_FRAME_LOCAL_NED,MAV_CMD_NAV_WAYPOINT,MAV_FRAME_GLOBAL_RELATIVE_ALT,MAV_FRAME_LOCAL_FRD,MAV_FRAME_BODY_NED
from coordinate_conversion import calculate_body_ned_coordinates


vehicle = connect_to_drone("tcp:localhost:5762")

print("Get some vehicle attribute values:")
print(" GPS: %s" % vehicle.gps_0)
print(" Battery: %s" % vehicle.battery)
print(" Last Heartbeat: %s" % vehicle.last_heartbeat)
print(" Is Armable?: %s" % vehicle.is_armable)
print(" System status: %s" % vehicle.system_status.state)
print(" Mode: %s" % vehicle.mode.name)    # settable

# print(vehicle.channels)

# while(True):
#     print(f"Roll : {degrees(vehicle._roll)}, Pitch : {degrees(vehicle._pitch)}")
#     time.sleep(0.5)

    # vehicle.channels.overrides['8'] = 1100
# print(vehicle.channels.overrides)
# vehicle.channels.overrides['15'] = 1100

print("sending")
# # vehicle.message_factory.command_int_send(0,0,MAV_FRAME_LOCAL_NED,MAV_CMD_DO_REPOSITION,0,0,4,0,0,0,10,5,-20)
# vehicle.message_factory.command_int_send(1,1,MAV_FRAME_GLOBAL,MAV_CMD_DO_REPOSITION,0,0,10,0,0,0,int(30.7481128 * 10**7), int(76.7565859 * 10**7),15)
x,y = calculate_body_ned_coordinates(vehicle.location.global_relative_frame.alt,vehicle.heading,0,-480)
print(x,y)
msg = vehicle.message_factory.set_position_target_local_ned_encode(
0,
0, 0,
MAV_FRAME_BODY_NED,  #relative to drone heading pos relative to EKF origin
0b0000000000000000,
x, y, 0,
0, 0,0,
0, 0, 0,
0, 0)
vehicle.send_mavlink(msg)

# print(vehicle.heading)
# print(vehicle._yaw)

time.sleep(1)
# Close vehicle object before exiting script
vehicle.close()