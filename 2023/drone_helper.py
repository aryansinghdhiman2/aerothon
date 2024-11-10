from dronekit import connect,VehicleMode,LocationGlobal,Vehicle, mavutil
import time
from simple_pid import PID
import pigpio
import math

GUIDED = VehicleMode("GUIDED")
AUTO = VehicleMode("AUTO")
RTL = VehicleMode('RTL')
LAND = VehicleMode('LAND')

PWM_PIN:int = 12
MIN_SERVO_PWM:int = 500
MAX_SERVO_PWM:int = 2500

DESCEND_SPEED:int = 1 #m/s 

hotspots:list[LocationGlobal] = []

def_address = 'tcp:localhost:5763'
#DRONE
def connect_to_drone(address:str) -> Vehicle:
    vehicle:Vehicle = connect(address)
    vehicle.wait_ready(True,timeout=300)
    return vehicle

def connect_to_drone_serial(address:str,baud:int) -> Vehicle:
    vehicle:Vehicle = connect(address,baud)
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
    return vehicle.location.global_relative_frame

def sendRollAndPitch(vehicle:Vehicle,roll_val:float,pitch_val:float,descend_speed:float=0) -> None:
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
    0,      
    0, 0,    
    mavutil.mavlink.MAV_FRAME_BODY_NED,  #relative to drone heading pos relative to EKF origin
    0b0000110111000111, 
    0, 0, 0,
    pitch_val, roll_val, descend_speed, 
    0, 0, 0, 
    0, 0)  
    vehicle.send_mavlink(msg)
#PID

class myController():
    def __init__(self,pid_roll:PID,pid_pitch:PID) -> None:
        self.roll = pid_roll
        self.pitch = pid_pitch

def configure_pid(p_roll:int,p_pitch:int,p_limit_roll:int,p_limit_pitch:int) -> myController:
    pid_roll = PID(p_roll,p_roll,0,setpoint=0)
    pid_pitch = PID(p_pitch,p_pitch,0,setpoint=0)
    
    pid_roll.output_limits = (-p_limit_roll,p_limit_roll)
    pid_pitch.output_limits = (-p_limit_pitch,p_limit_pitch)

    controller = myController(pid_roll,pid_pitch)
    return controller

def move_to_center(vehicle:Vehicle, controller:myController,current_x:float,current_y:float,descend_speed:float=0) -> None:
    delta_x = current_x
    delta_y = current_y

    roll = controller.roll(delta_x)
    pitch = controller.pitch(delta_y)
    print(f"X:{current_x}, Y:{current_y}, DX:{delta_x}, DY:{delta_y}, ROLL:{roll}, PITCH: {pitch}")
    sendRollAndPitch(vehicle,roll,pitch,descend_speed)
    
def setGripper(pi) -> None:
    pi.set_servo_pulsewidth(PWM_PIN,MIN_SERVO_PWM)
    
def releaseGripper(pi) -> None:
    pi.set_servo_pulsewidth(PWM_PIN,MAX_SERVO_PWM)

def moveToAlt(vehicle:Vehicle,lat:float,lon:float,alt:int) -> None:
    print(f"Coming down {alt}")
    msg = vehicle.message_factory.set_position_target_global_int_encode(
    0,
    0, 0,
    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,  #relative to drone heading pos relative to EKF origin
    0b0000110111111000,
    int(lat * 10**7), int(lon * 10**7), alt,
    0, 0,0,
    0, 0, 0,
    0, 0)
    vehicle.send_mavlink(msg)

def descendAndTakePhoto(vehicle:Vehicle,controller:myController,delta_x:int,delta_y:int,cv,frame) -> bool:
    current_location = getCurrentLocation(vehicle)
    print(getCurrentLocation(vehicle).alt)
    desc_speed = 0 if int(getCurrentLocation(vehicle).alt) < 11 else DESCEND_SPEED
    delta_x = delta_x if abs(delta_x) > 25 else 0
    delta_y = delta_y if abs(delta_y) > 25 else 0
    if(abs(delta_x) > 25 or abs(delta_y) > 25 ):
        move_to_center(vehicle,controller,delta_x,delta_y,0)
        print(f"MOVING TO CENTER {desc_speed}")
        return False
    else:
        print("MOving to 10")
        moveToAlt(vehicle,current_location.lat,current_location.lon,10);
        while(getCurrentLocation(vehicle).alt > 11):
            time.sleep(1)
        #TAKE PHOTO
        cv.imwrite(f"/home/pi/{time.ctime(time.time())}_captured_image.jpg", frame)
        print("Taken PHOTO")
        #SAVE COORDINATES
        hotspots.append(getCurrentLocation(vehicle))
        print("MOving to 30")
        moveToAlt(vehicle,current_location.lat,current_location.lon,30);
        while(getCurrentLocation(vehicle).alt < 19):
            time.sleep(1)
        return True

def descendAndRelease(vehicle:Vehicle,controller:myController,delta_x:int,delta_y:int,pi):
    vehicle.mode = GUIDED
    vehicle.wait_for_mode(GUIDED)
    current_location = getCurrentLocation(vehicle)
    print("Descending rel")
    if(abs(delta_x) > 25 or abs(delta_y) > 25 or getCurrentLocation(vehicle).alt > 21):
        move_to_center(vehicle,controller,delta_x,delta_y,0)
    else:
        print("MOving to 20")
        moveToAlt(vehicle,current_location.lat,current_location.lon,20);
        while(getCurrentLocation(vehicle).alt > 21):
            time.sleep(1)
        #TAKE PHOT
        time.sleep(2)
        print('release')
        releaseGripper(pi)
        #SAVE COORDINATES
        hotspots.append(getCurrentLocation(vehicle))
        print("MOving to 30")
        moveToAlt(vehicle,current_location.lat,current_location.lon,30);
        while(getCurrentLocation(vehicle).alt < 19):
            time.sleep(1)

def get_distance_metres(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.

    This method is an approximation, and will not be accurate over large distances and close to the 
    earth's poles. It comes from the ArduPilot test code: 
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5
