from dronekit import connect,VehicleMode,LocationGlobal,Vehicle, mavutil
import time
from drone_helper import getCurrentLocation,GUIDED,AUTO,moveToAlt

from coordinate_conversion import calculate_gps_coordinates

def move_to_center_image_coords(vehicle:Vehicle,x:int,y:int) -> None:
    current_location = getCurrentLocation(vehicle)

    p_lat,p_lon = calculate_gps_coordinates(current_location.lat,current_location.lon,current_location.alt,5.1,960,720,int(x),int(y),vehicle.heading)
    predicted_coords = LocationGlobal(p_lat,p_lon)

    moveToAlt(vehicle,p_lat,p_lon,5)
    # vehicle.simple_goto(predicted_coords)


def move_to_center_image_gps(vehicle:Vehicle,location:LocationGlobal) -> None:
    
    moveToAlt(vehicle,location.lat,location.lon,5)
    # vehicle.simple_goto(predicted_coords)


def descendAndReleaseGPS(vehicle:Vehicle,x:int,y:int) -> None:
    vehicle.mode = GUIDED
    vehicle.wait_for_mode(GUIDED)
    print("Descending rel")
    
    move_to_center_image_coords(vehicle,x,y)
    
    while(getCurrentLocation(vehicle).alt > 6):
        time.sleep(1)
    
    #TAKE PHOTO
    time.sleep(2)
    print('release')
    #SAVE COORDINATES


    print("Moving to 15")
    current_location = getCurrentLocation(vehicle)
    moveToAlt(vehicle,current_location.lat,current_location.lon,15)
    while(getCurrentLocation(vehicle).alt < 13):
        time.sleep(1)

def descendAndTakePhoto(vehicle:Vehicle,x:int,y:int) -> None:
    current_location = getCurrentLocation(vehicle)
    print(getCurrentLocation(vehicle).alt)
    
    move_to_center_image_coords(vehicle,x,y)

    while(getCurrentLocation(vehicle).alt > 11):
        time.sleep(1)
    
    #TAKE PHOTO
    print("Taken PHOTO")
    #SAVE COORDINATES
