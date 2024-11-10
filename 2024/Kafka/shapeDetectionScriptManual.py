import socketio
import time
import asyncio
from ultralytics import YOLO
import json
import cv2
import threading
from drone_helper import connect_to_drone, goto_center, AUTO, GUIDED, Vehicle, align_at_center, drop_and_return_to_15
from classificationEnum import TARGET
from classificationEnum import HOTSPOT, TARGET, DET_OBJ
from drone_helper import connect_to_drone, getCurrentLocation
from socketio.exceptions import TimeoutError
import torch
import numpy as np

channel_6 = 0


# vehicle: Vehicle = connect_to_drone("tcp:172.24.240.1:5762")
vehicle: Vehicle = connect_to_drone("tcp:10.42.0.1:14550")
@vehicle.on_message("RC_CHANNELS")
def channel_handler(vehicle:Vehicle,name,message):
    global channel_6
    # print("Name ",name)
    # print("Message: ",message.chan6_raw)
    channel_6 = message.chan6_raw

MODEL_NAMES = ["./OptimizedWeights/best_shapes.pt"]
# SOURCES = ["../../../../videos/New Project - Made with Clipchamp.mp4"]
SOURCES = ["./output16.avi"]


async def run_tracker_in_thread(model_name, filename):
    
    frame_cnt = 0
    # cam = cv2.VideoCapture("rtspsrc location=rtsp://10.42.0.1:8554/cam latency=0 protocols=tcp ! decodebin ! videoconvert ! appsink drop=1 max-buffers=5 max-bytes=1843488", cv2.CAP_GSTREAMER)
    cam = cv2.VideoCapture(
        "rtspsrc location=rtsp://172.24.240.1:8554/cam latency=0 protocols=tcp ! decodebin ! videoconvert ! appsink drop=1 max-buffers=5 max-bytes=1843488", cv2.CAP_GSTREAMER)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./TestResults/output_71_shape.avi',
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
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break

        results = model(frame, conf=0.50, iou=0.70)
        annotated_frame = results[0].plot()
        for r in results:
            await asyncio.sleep(0)
            print(f"Frame count - {frame_cnt}")
            print("CHANNEL 6:",channel_6)

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
                    if label == 0:
                        json_obj = {
                            "center": center,
                            # "location": location,
                            "type": HOTSPOT
                        }
                        print("Hotspot found")
                        annotated_frame=frame
                    if label == 2:
                        annotated_frame = frame
                if(channel_6 >= 1800):
                    print("Starting opencv det")
                    annotated_frame = frame                        
                    annotated_frame = cv2.resize(
                        annotated_frame, (int(annotated_frame.shape[1]), int(annotated_frame.shape[0])))
                    circles = 0
                    squares = 0
                    triangles = 0
                    color = (255, 255, 0)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    gray_image = cv2.cvtColor(
                        annotated_frame, cv2.COLOR_BGR2GRAY)
                    gray_image = cv2.blur(gray_image, (7, 7))
                    kernel1=np.ones((1,1),np.uint8)
                    kernel2=np.ones((5,5),np.uint8)
                    gray_image =cv2.erode(gray_image,kernel1,iterations=1)
                    gray_image=cv2.dilate(gray_image,kernel1,iterations=1)
                    _, thresh_image = cv2.threshold(
                        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contours, hierarchy = cv2.findContours(
                        thresh_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(
                        annotated_frame, contours, -1, (0, 255, 0), 3)
                    if hierarchy is not None:
                        # Flatten the hierarchy
                        hierarchy = hierarchy[0]
                    text = ""
                    for i, contour in enumerate(contours):
                        # Check if the contour is a child (i.e., it has a parent contour)
                        # Parent != -1 means it has a parent
                        if hierarchy[i][3] != -1:
                            epsilon = 0.07 * \
                                cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(
                                contour, epsilon, True)

                            # Get the bounding box coordinates
                            x, y, w, h = cv2.boundingRect(
                                contour)

                            cv2.rectangle(
                                annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            if len(approx) == 3:
                                triangles += 1
                                cv2.putText(annotated_frame, 'Triangle', (x, y - 10), font,
                                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            elif len(approx) == 4:
                                squares += 1
                                cv2.putText(annotated_frame, 'Square', (x, y - 10), font,
                                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            else:
                                circles += 1
                                text = f"Triangle count = {triangles}\nSquare count = {squares}\nTotal shape count = {triangles + squares}"
                    font_scale = 0.7
                    color = (0, 255, 255)
                    thickness = 2
                    line_type = cv2.LINE_AA

                    lines = text.split('\n')

                    (height, width) = annotated_frame.shape[:2]

                    y0, dy = height - 20, 30

                    for i, line in enumerate(lines):
                        y = y0 - i * dy * 2
                        (text_width, text_height), baseline = cv2.getTextSize(
                            line, font, font_scale, thickness)
                        x = width - text_width - 60
                        cv2.putText(
                            annotated_frame, line, (x, y), font, font_scale, color, thickness, line_type)
                    with open("./Shapecount.jsonl","a") as outfile_2:
                        json.dump({
                            "square_count":squares,
                            "triangle_cnt":triangles,
                            "total_cnt":int(squares+triangles)
                        },outfile_2)
                        outfile_2.write("\n")
                                

            out.write(annotated_frame)
            cv2.imshow("Detections",annotated_frame)
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
