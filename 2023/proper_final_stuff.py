import cv2 as cv
import numpy as np
from math import pi
from itertools import combinations
import time
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder, Quality
import cv2 as cv
import numpy as np
from drone_helper import *
import pigpio
from coordinate_conversion import calculate_gps_coordinates

# detected_item = 0  # 0 none, 1 hotspot, 2 targetq


def distance(coord1, coord2):
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5

def priority(item):
    return distance(item['center'],(480,360))



def find_close_coordinates_and_average(centers, threshold):
    grouped_coordinates = []

    for coord1, coord2 in combinations(centers, 2):
        if distance(coord1["center"], coord2["center"]) < threshold:
            merged = False
            for group in grouped_coordinates:
                if coord1 in group or coord2 in group:
                    group.append(coord1)
                    group.append(coord2)
                    merged = True
                    break

            if not merged:
                grouped_coordinates.append([coord1, coord2])

    # Remove duplicates within each group
    for group in grouped_coordinates:
        group[:] = [dict(t) for t in {tuple(d.items()) for d in group}]

    averaged_coordinates = []

    for group in grouped_coordinates:
        x_avg = sum(coord["center"][0] for coord in group) / len(group)
        y_avg = sum(coord["center"][1] for coord in group) / len(group)
        averaged_coordinates.append({
            "center": (x_avg, y_avg),
            "color": "hotspot" if all(item["color"] == "red" for item in group) else "target"
        })
    averaged_coordinates.sort(key = priority)
    return averaged_coordinates


def red_detection(frame, hsv_frame):
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    morph1 = cv.dilate(red_mask, np.ones((3, 3), np.uint8), iterations=2)
    morph2 = cv.morphologyEx(morph1, cv.MORPH_OPEN, kernel)
    morph_final = cv.morphologyEx(morph2, cv.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8))

    contours, hierarchy = cv.findContours(
        morph_final, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    centers_list = []
    for contour in contours:
        if cv.contourArea(contour)>600:
            center, radius = cv.minEnclosingCircle(contour)
            x, y = center
            cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 4)
            centers_list.append({"center": (int(x), int(y)), "color": "red"})

    #cv.imshow("Red Objects in Video1", cv.resize(morph_final, (640, 360)))
    return centers_list  # list of red minEnclosing circle centers


def blue_detection(frame, hsv_frame, gray_frame):
    lower_blue = np.array([100, 109, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv.inRange(hsv_frame, lower_blue, upper_blue)
    red_mask = cv.bitwise_and(gray_frame, gray_frame, mask=mask)
    _, red_mask = cv.threshold(red_mask, 25, 255, cv.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    morph1 = cv.dilate(red_mask, np.ones((3, 3), np.uint8), iterations=1)
    morph2 = cv.morphologyEx(morph1, cv.MORPH_OPEN, kernel)
    morph_final = cv.morphologyEx(morph2, cv.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8))

    # contours
    contours, hierarchy = cv.findContours(
        morph_final, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # getting contours with children and child contours

    centers_list = []
    for contour in contours:
        if cv.contourArea(contour)>600:
            center, radius = cv.minEnclosingCircle(contour)
            x, y = center
            cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 4)
            centers_list.append({"center": (int(x), int(y)), "color": "blue"})

    #cv.imshow("Blue Objects in Video1", cv.resize(morph_final, (640, 360)))
    return centers_list  # list of blue contoour minEnclosing circle centers


#video_capture = cv.VideoCapture('images\cut_vlc.mp4')

cam = Picamera2()
config= cam.create_video_configuration(main={"size": (960,720), "format": "RGB888"}, controls= { "FrameDurationLimits": (150000,150000)})
cam.configure(config)

timestamp = time.ctime(time.time())

#load the video
#video_recorded = cv.VideoCapture('../Sun Nov 12 10:47:00 2023-detection-trials.avi')

fourcc = cv.VideoWriter_fourcc(*"XVID")
out = cv.VideoWriter(f"/home/pi/{timestamp}-detection-trials.avi", fourcc, 6, (960,720))

#cam.start_preview(Preview.QTGL)
cam.start()
cam.set_controls({"AfMode": 2, "AfRange": 0,"AfSpeed": 1})
print("Connecting to drone")
vehicle = connect_to_drone_serial("/dev/ttyACM0",57600)
#vehicle = connect_to_drone('tcp:192.168.242.47:5762')
pi = pigpio.pi()
print("Connected")
@vehicle.on_message('RC_CHANNELS')
def chin_listener(self, name, message):
 gripper=message.chan7_raw  
 if(gripper > 1700):
     pi.set_servo_pulsewidth(12,2500)
 if(gripper < 1400):
     pi.set_servo_pulsewidth(12,500)

while True:
    frame = cam.capture_array()
    pic_frame = frame.copy()
    #read the loaded video
    #_,frame = video_recorded.read()
    #pic_frame = frame.copy()
    frame = cv.rotate(frame, cv.ROTATE_180)
    frame_width , frame_height = (960, 720)
    frame_center = (frame_width //2 , frame_height //2)
    print(frame_center)
    cv.circle(frame, frame_center, 3, (0,255,0), -1)

    blurred_frame = cv.GaussianBlur(frame, (5, 5), 9, 9)
    gray_frame = cv.cvtColor(blurred_frame, cv.COLOR_BGR2GRAY)
    hsv_frame = cv.cvtColor(blurred_frame, cv.COLOR_BGR2HSV)

    # returns the list of centres of the circles with the RED mask
    center_red = red_detection(frame, hsv_frame)
    # returns the list of centers of the circles with the BLUE mask
    center_blue = blue_detection(frame, hsv_frame, gray_frame)

    # Grouping done here...the groups must also contain the label (hotspot vs target)
    result = find_close_coordinates_and_average(center_blue + center_red, 3)
    detected = False
    if result:
        for data in result:
            x, y = data["center"]
            cv.circle(frame, (int(x), int(y)), 5, (255, 255, 255), -1)
            text_color = (0, 255, 255)
            cv.putText(frame, f"{data['color']} detected: {(int(x), int(y))}", (int(
                x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, text_color, 3, cv.LINE_AA)
            
            near_a_hotspot = False
            for hotspot in hotspots:
                current_location = getCurrentLocation(vehicle)
                p_lat,p_lon = calculate_gps_coordinates(current_location.lat,current_location.lon,current_location.alt,5.1,960,720,int(x),int(y),vehicle.heading)
                predicted_coords = LocationGlobal(p_lat,p_lon)
                dist = get_distance_metres(hotspot,predicted_coords)
                if(dist < 10):
                    print(f"HOTSPOT DETECTED AT img: {x},{y} gps:{predicted_coords.lat},{predicted_coords.lon}")
                    near_a_hotspot = True
                    break
            if(near_a_hotspot): continue
            
            
            if((vehicle.mode == GUIDED or vehicle.mode == AUTO)):
                vehicle.mode = GUIDED
                vehicle.wait_for_mode(GUIDED)
                controller = configure_pid(-0.00075,-0.00075,1,1)
                adjusted_center = (x-(960/2),-(y-(720/2)))
                print(f"center: {x},{y} adj:{adjusted_center}")
                if(data['color'] == 'hotspot'):
                    descendAndTakePhoto(vehicle,controller,adjusted_center[0],adjusted_center[1],cv,pic_frame)
                    pass
                if(data['color'] == 'target'):
                    descendAndRelease(vehicle,controller,adjusted_center[0],adjusted_center[1],pi)
                    pass
            detected = True

    #cv.imshow("Final video", frame)
    if(detected == False):
        if(vehicle.mode == GUIDED):
            vehicle.mode = AUTO
            
    detected = False
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    #Save Video
    out.write(frame)
    print(hotspots)
    
video_capture.release()
cv.destroyAllWindows()
out.release()
