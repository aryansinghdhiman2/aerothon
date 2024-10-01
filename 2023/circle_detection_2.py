import time
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder, Quality
import cv2 as cv
import numpy as np
from drone_helper import *
import pigpio
from coordinate_conversion import calculate_gps_coordinates

cam = Picamera2()
config= cam.create_video_configuration(main={"size": (960,720), "format": "RGB888"}, controls= { "FrameDurationLimits": (150000,150000)})
cam.configure(config)

timestamp = time.ctime(time.time())

#load the video
#video_recorded = cv.VideoCapture('../Thu Nov  9 13:32:33 2023-detection-trials.avi')

fourcc = cv.VideoWriter_fourcc(*"XVID")
out = cv.VideoWriter(f"/home/pi/{timestamp}-detection-trials.avi", fourcc, 6, (960,720))

#cam.start_preview(Preview.QTGL)
cam.start()
cam.set_controls({"AfMode": 2, "AfRange": 0,"AfSpeed": 1})
print("Connecting to drone")

#vehicle = connect_to_drone('tcp:172.16.194.58:5762')
#vehicle = connect_to_drone('tcp:192.168.242.47:5762')
vehicle = connect_to_drone_serial("/dev/ttyACM0",57600)
pi = pigpio.pi()
print("Connected")
@vehicle.on_message('RC_CHANNELS')
def chin_listener(self, name, message):
 # print '%s attribute is: %s' % (name, message)
 gripper=message.chan7_raw  
 #print(f'channels:{message}')
 if(gripper > 1700):
     pi.set_servo_pulsewidth(12,2500)
 if(gripper < 1400):
     pi.set_servo_pulsewidth(12,500)

while(vehicle.mode != AUTO):
    time.sleep(1)

while True:
    frame = cam.capture_array()
    #read the loaded video
    #_,frame = video_recorded.read()
    frame = cv.rotate(frame, cv.ROTATE_180)
    frame_width , frame_height = (960, 720)
    frame_center = (frame_width //2 , frame_height //2)
    print(frame_center)
    # Convert the frame to the HSV color space
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define the HSV range for red color
    lower_red1 = np.array([0, 100, 100])  # Lower bound of hue, saturation, and value
    upper_red1 = np.array([10, 255, 255])  # Upper bound of hue, saturation, and value

    lower_red2 = np.array([160, 100, 100])  # Lower bound of hue, saturation, and value
    upper_red2 = np.array([180, 255, 255])  # Upper bound of hue, saturation, and value

    # Create masks for both red color ranges and combine them
    mask1 = cv.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv.bitwise_or(mask1, mask2)

    # Find contours in the red mask
    contours, _ = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables for the largest circle
    largest_circle = None
    largest_radius = 0

    # Filter out smaller circles and find the largest one
    for contour in contours:
        # Calculate area of the circle
        area = cv.contourArea(contour)
        
        # Check if the area is 1000 or more
        if area >= 650:
            # Calculate center and radius of the cir    cle
            (x, y), radius = cv.minEnclosingCircle(contour)
            
            if radius > largest_radius:
                largest_radius = radius
                largest_circle = contour
    detected:bool = False
    near_a_hotspot:bool = False
    if largest_circle is not None:
        print("DETECTED")
        # Calculate center and radius of the largest circle
        detected = True
        near_a_hotspot = False
        (x, y), radius = cv.minEnclosingCircle(largest_circle)
        center = (int(x), int(y))
        radius = int(radius)

        # Draw the largest circle
        cv.circle(frame, center, radius, (0, 255, 0), 2)

        # Mark the center
        cv.circle(frame, center, 5, (0, 0, 255), -1)  # Red center marker
        cv.circle(frame, frame_center, 3, (0,255,0), -1)    
        # Annotate the largest circle with the center coordinates
        text = f"Largest Circle Center: ({x:.2f}, {y:.2f})"
        cv.putText(frame, text, (center[0] - 50, center[1] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        #Drone control
        adjusted_center = (center[0]-(960/2),-(center[1]-(720/2)))
        print(f"center: {center}, adj :{adjusted_center}")
        current_location:LocationGlobal = getCurrentLocation(vehicle)
        print(current_location.lat,current_location.lon,current_location.alt)
        
        
        for hotspot in hotspots:
            p_lat,p_lon = calculate_gps_coordinates(current_location.lat,current_location.lon,current_location.alt,5.1,960,720,center[0],center[1],vehicle.heading)
            predicted_coords = LocationGlobal(p_lat,p_lon)
            dist = get_distance_metres(hotspot,predicted_coords)
            if(dist < 10):
                print(f"HOTSPOT DETECTED AT img: {center} gps:{predicted_coords.lat},{predicted_coords.lon}")
                near_a_hotspot = True
                break
            
        photo_taken = False
        if((vehicle.mode == GUIDED or vehicle.mode == AUTO) and not near_a_hotspot):
            vehicle.mode = GUIDED
            vehicle.wait_for_mode(GUIDED)
            #controller = configure_pid(-0.2,-0.2,3,3)
            controller = configure_pid(-0.0015,-0.0015,1,1)
            print('DESCENDING')
            photo_taken = descendAndTakePhoto(vehicle,controller,adjusted_center[0],adjusted_center[1],cv,frame)
    
    # Display the result
    if(detected == False or (detected == True and near_a_hotspot)):
        print('ATTEMPTING TO CHANGE')
        print(detected)
        print(near_a_hotspot)
        
        if(vehicle.mode == GUIDED):
            vehicle.mode = AUTO
    print(hotspots)
    # Start Preview
    #cv.imshow("Red Objects in Video", cv.resize(frame,(700,400)))
    if cv.waitKey(1)==ord('q'):
        break
  
    #Save Video
    out.write(frame)
    
cv.destroyAllWindows()
out.release()
#vehicle.close()
