import socketio
import time
import asyncio
from ultralytics import YOLO
import json
import cv2
import threading
import json
import numpy as np
from drone_helper import connect_to_drone, goto_center, AUTO, GUIDED, Vehicle, align_at_center, drop_and_return_to_15
from classificationEnum import TARGET
from classificationEnum import HOTSPOT, TARGET, DET_OBJ
from drone_helper import connect_to_drone, getCurrentLocation
from socketio.exceptions import TimeoutError
import torch

alignment_flag = [True, False, False, False, False]
alignment_state = 0
# 0, 1, 2
min_area = 900  # Adjust as needed
distance_threshold = 15  # Tolerance for centroid matching
sio = socketio.AsyncClient()


vehicle: Vehicle = connect_to_drone("udpout:10.42.0.1:10000")

async def start_server():
    await sio.connect('http://172.24.240.1:5000')
    print("Connected to server")


@sio.on('requested_alignment_state')
def handler(msg):
    print('received event:', "requested_alignment_state", msg)
    global alignment_state
    global alignment_flag
    alignment_state = msg
    # target_or_shape = msg.target_or_shape
    alignment_flag[alignment_state] = True
    # if (target_or_shape == "Shape"):
    #     print("SCRIPT HAS TO RUN AFTER ALIGNMENT")

async def emit_alignment(alignment_state, center):
    print("EMIT ALIGNMENT CALLED")
    await sio.emit("alignment", {"alignment_state": alignment_state,
                                "center": center})
    print("EMIT ENDED")


MODEL_NAMES = ["./OptimizedWeights/hyper_param_tuned.pt"]
# SOURCES = ["../../../../videos/New Project - Made with Clipchamp.mp4"]
SOURCES = ["./output16.avi"]


async def run_tracker_in_thread(model_name, filename):
    await start_server()
    global alignment_state
    global alignment_flag
    
    frame_cnt = 0
    # cam = cv2.VideoCapture("rtspsrc location=rtsp://10.42.0.1:8554/cam latency=0 protocols=tcp ! decodebin ! videoconvert ! appsink drop=1 max-buffers=5 max-bytes=1843488", cv2.CAP_GSTREAMER)
    cam = cv2.VideoCapture(
        "rtspsrc location=rtsp://172.24.240.1:8554/cam latency=0 protocols=tcp ! decodebin ! videoconvert ! appsink drop=1 max-buffers=5 max-bytes=1843488", cv2.CAP_GSTREAMER)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./TestResults/output_77.avi',
                          fourcc, 20.0, (640,  480))
    # cam = cv2.VideoCapture(
    #     "filesrc location=/mnt/c/Users/91798/Desktop/test_yolo/aerothon_repo/aerothon/2024/Kafka/output16.avi ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    time.sleep(3)
    model = YOLO(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    if not cam.isOpened():
        print("Error: Cannot open RTSP stream.")
        exit()

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # target_alignment_flag = False
    # Shape_alignment_flag = False
    min_area = 500

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break

        results = model(frame, conf=0.60, iou=0.70)
        annotated_frame = results[0].plot()
        for r in results:
            await asyncio.sleep(0)
            locationObj = getCurrentLocation(vehicle)
            location = [locationObj.lat, locationObj.lon,
                        locationObj.alt, vehicle.heading]
            
            print(f"Frame count - {frame_cnt}")
            print("ALIGNMENT STATE == ",alignment_state)

            frame_cnt += 1
            obj = (json.loads(r.to_json()))
            im_array = r.plot()
            conv_obj = {}
            if len(obj) > 0:
                conv_obj = {
                    "boxes": [obj[0]['box']],
                    "labels": [obj[0]['class']],
                    "scores": [obj[0]['confidence']]
                }
            else:
                conv_obj = {
                    "boxes": [],
                    "labels": "none",
                    "scores": []
                }
            draw_box = (0, 0)
            if len(conv_obj["scores"]) > 0:
                for box in conv_obj["boxes"]:
                    print(box)
                    x1 = box["x1"]
                    x2 = box["x2"]
                    y1 = box["y1"]
                    y2 = box["y2"]
                    draw_box = (((x1+x2)/2), ((y1+y2)/2))
            cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
            cv2.putText(annotated_frame, f"({int(draw_box[0]-320)},{int(240-draw_box[1])})", (int(draw_box[0]), int(
                draw_box[1])), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, lineType=cv2.LINE_AA, color=(0, 255, 0))
            with open("./json_output.jsonl", "a") as outfile:
                res = conv_obj
                if len(res["scores"]) == 0:
                    json.dump({
                        "box": "None",
                        "label": "None",
                        "predicted_confidence_score": "None",
                        "location": []
                    }, outfile)
                    outfile.write('\n')
                print("155 line printing")
                for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                    print("BOXXXX",box,type(box))
                    box = (box['x1'], box['y1'],
                           box['x2'], box['y2'])
                    print('159 line printed')
                    x_min, y_min, x_max, y_max = box
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2

                    center = (center_x, center_y)
                    predicted_label = label
                    predicted_confidence_score = score
                    print("LINE 167")
                    json.dump({
                        "box": box,
                        "label": predicted_label,
                        "predicted_confidence_score": predicted_confidence_score,
                        "center": center
                    }, outfile)
                    outfile.write('\n')
                    print("174 line")
                    print({center_x,center_y},"ASIDNUASNDIUASNDIASND")
                    if label == 1:
                        json_obj = {
                            "center": center,
                            # "location": location,
                            "type": HOTSPOT
                        }
                        print("Hotspot found")
                        image = cv2.resize(image, (int(image.shape[1] * 0.7), int(image.shape[0] * 0.7)))
                    new_image = image.copy()

                    font = cv2.FONT_HERSHEY_SIMPLEX

                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    blurred_gray_image = cv2.blur(gray_image, (7, 7))
                    
                    _, threshed_image = cv2.threshold(blurred_gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Morphological transformations
                    kernel1 = np.ones((1, 1), np.uint8)
                    eroded_image = cv2.erode(threshed_image, kernel1, iterations=1)
                    dilated_image = cv2.dilate(eroded_image, kernel1, iterations=1)

                    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    hsv_frame = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
                    lower_red = np.array([141, 77, 129])
                    upper_red = np.array([255, 255, 255])
                    mask = cv2.inRange(hsv_frame, lower_red, upper_red)
                    result = cv2.bitwise_and(new_image, new_image, mask=mask)

                    mask_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    def get_centroid(contour):
                        M = cv2.moments(contour)
                        if M["m00"] == 0:
                            return None
                        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    triangles = 0
                    squares = 0
                    color = (0, 255, 0)

                    parent_shapes = {}

                    for i, contour in enumerate(contours):
                        if cv2.contourArea(contour) > min_area:
                            epsilon = 0.07 * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            shape_type = None
                            if len(approx) == 3:
                                shape_type = "Triangle"
                            elif len(approx) == 4:
                                shape_type = "Square"
                            else:
                                shape_type = "Circle"

                            centroid = get_centroid(contour)
                            if centroid is None:
                                continue

                            for mask_contour in mask_contours:
                                if cv2.contourArea(mask_contour) > min_area:
                                    mask_centroid = get_centroid(mask_contour)
                                    if mask_centroid is None:
                                        continue

                                    distance = np.sqrt((centroid[0] - mask_centroid[0]) ** 2 + (centroid[1] - mask_centroid[1]) ** 2)
                                    if distance < distance_threshold:
                                        if shape_type == "Triangle":
                                            triangles += 1
                                        elif shape_type == "Square":
                                            squares += 1
                                        
                                        # Draw and label shape on the image
                                        cv2.putText(image, shape_type, (approx[0][0][0], approx[0][0][1] - 10), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                                        cv2.drawContours(image, [contour], -1, (0, 0, 0), 1)
                                        x, y, w, h = cv2.boundingRect(contour)
                                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                                        parent_idx = hierarchy[0][i][3]
                                        if parent_idx not in parent_shapes:
                                            parent_shapes[parent_idx] = []
                                        parent_shapes[parent_idx].append(shape_type)
                    center = (0,0)
                    for parent_idx, shapes in parent_shapes.items():
                        if len(shapes) > 1:  
                            merged_contour = contours[parent_idx]
                            cv2.drawContours(image, [merged_contour], -1, (255, 255, 0), 2) 
                            (center_x, center_y), radius = cv2.minEnclosingCircle(merged_contour)
                            center = (int(center_x), int(center_y))
                            radius = int(radius)
                            cv2.circle(image, center, radius, (255, 255, 0), 2) 
                            cv2.circle(image, center, 5, (0, 0, 255), -1) 
                    if(center[0] != 0 and center[1] != 0):
                        adjusted_center = [center[0]-360,240-center[1]]
                        if (alignment_state <= 2):
                            if (vehicle.mode == AUTO or vehicle.mode == GUIDED):
                                print("VEHICLE MODE", vehicle.mode)
                                if (alignment_flag[0]):
                                    print("sent 0")
                                    await emit_alignment(alignment_state,
                                                            location, adjusted_center)
                                    alignment_flag[0] = False
                                elif (alignment_flag[1]):
                                    print("sent 1")
                                    await emit_alignment(alignment_state,
                                                            location, adjusted_center)

                                elif (alignment_flag[2]):
                                    alignment_flag[1] = False
                                    print("sent 2")
                                    with open("./Shapecount.jsonl","a") as outfile_2:
                                      json.dump({
                                          "square_count":squares,
                                          "triangle_cnt":triangles,
                                          "total_cnt":int(squares+triangles)
                                      },outfile_2)
                                      outfile_2.write("\n")
                                    alignment_flag[2] = False

                                             
            cv2.imshow("Detections",annotated_frame)
            out.write(annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    pass


async def main():
    for video_file, model_name in zip(SOURCES, MODEL_NAMES):
        await run_tracker_in_thread(model_name, video_file)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
