from dronekit import connect,VehicleMode,LocationGlobal,Vehicle, mavutil
import time
from simple_pid import PID
GUIDED = VehicleMode("GUIDED")
AUTO = VehicleMode("AUTO")
RTL = VehicleMode('RTL')

def_address = 'tcp:localhost:5763'
#DRONE
def connect_to_drone(address:str) -> Vehicle:
    vehicle:Vehicle = connect(address)
    vehicle.wait_ready(True,timeout=300)
    return vehicle

def arm(vehicle:Vehicle) -> None:
    while not vehicle.is_armable:
        print("Waiting for drone to be in armable state")
        time.sleep(1)
    vehicle.mode = GUIDED
    vehicle.armed = True
    while not vehicle.armed:
        print("Arming")
        time.sleep(1)

def start_mission(vehicle:Vehicle) -> None:
    vehicle.commands.download()
    vehicle.commands.wait_ready()

    vehicle.commands.next = 0
    vehicle.mode = AUTO
    vehicle.wait_for_mode(AUTO)
    print('Starting Mission')

def resume_mission(vehicle:Vehicle) -> None:
    vehicle.mode = AUTO
    vehicle.wait_for_mode(AUTO)

def getCurrentLocation(vehicle:Vehicle) -> LocationGlobal:
    return vehicle.location.global_frame

def sendRollAndPitch(vehicle:Vehicle,roll_val:float,pitch_val:float) -> None:
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
    0,      
    0, 0,    
    mavutil.mavlink.MAV_FRAME_BODY_NED,  #relative to drone heading pos relative to EKF origin
    0b0000110111000111, #ignore velocity z and other pos arguments
    0, 0, 0,
    roll_val, pitch_val, 0, 
    0, 0, 0, 
    0, 0)  
    vehicle.send_mavlink(msg)
#PID

class myController():
    def __init__(self,pid_roll:PID,pid_pitch:PID) -> None:
        self.roll = pid_roll
        self.pitch = pid_pitch

def configure_pid(p_roll:int,p_pitch:int,p_limit_roll:int,p_limit_pitch:int) -> myController:
    pid_roll = PID(p_roll,0,0,setpoint=0)
    pid_pitch = PID(p_pitch,0,0,setpoint=0)
    
    pid_roll.output_limits = (-p_limit_roll,p_limit_roll)
    pid_pitch.output_limits = (-p_limit_pitch,p_limit_pitch)

    controller = myController(pid_roll,pid_pitch)
    return controller

def move_to_center(vehicle:Vehicle, controller:myController,current_x:float,current_y:float) -> None:
    delta_x = current_x
    delta_y = current_y

    roll = controller.roll(delta_x)
    pitch = controller.pitch(delta_y)

    sendRollAndPitch(vehicle,roll,pitch)