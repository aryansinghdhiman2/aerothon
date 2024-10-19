import cv2
import random
import json
import torch
from drone_helper import connect_to_drone, descendAndReleaseImg, LocationGlobal, get_distance_metres, AUTO, GUIDED, Vehicle
from classificationEnum import TARGET
import os
from classificationEnum import HOTSPOT, TARGET, DET_OBJ
from drone_helper import connect_to_drone, getCurrentLocation

from ultralytics import YOLO

vehicle: Vehicle = connect_to_drone("tcp:localhost:5762")

found_target = False


def draw_boxes(image, results):
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = (box['x1'], box['y1'], box['x2'], box['y2'])
        predicted_label = label
        predicted_confidence_score = score

        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        color = (0, 255, 0)

        thickness = 2
        cv2.rectangle(image, start_point, end_point, color, thickness)

        label_text = f"""Label: {predicted_label}, Score: {
            predicted_confidence_score:.2f}"""
        cv2.putText(image, label_text, start_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    return image


model = YOLO('../../../../YoloNew/best.pt')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(device)
frame_no = 0
video_path = '../../../../videos/output.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = './detections/output_video_test.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    locationObj = getCurrentLocation(vehicle)
    location = [locationObj.lat, locationObj.lon,
                locationObj.alt, vehicle.heading]
    if not ret:
        break

    results = model(frame, conf=0.7, iou=0.6)
    r = results[0]
    conv_obj = {}
    obj = (json.loads(results[0].to_json()))
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
    print(type(results), results)
    results = conv_obj
    frame_no += 1
    center = (0, 0)
    with open("./json_output.jsonl", "a") as outfile:
        if len(results["scores"]) == 0:
            json.dump({
                "box": "None",
                "frame_no": frame_no,
                "video_name": video_path,
                "label": "None",
                "predicted_confidence_score": "None",
                "location": []
            }, outfile)
            outfile.write('\n')

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = (box['x1'], box['y1'],
                   box['x2'], box['y2'])

            x_min, y_min, x_max, y_max = box
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            center = (center_x, center_y)
            predicted_label = label
            predicted_confidence_score = score
            json.dump({
                "box": box,
                "frame_no": frame_no,
                "video_name": video_path,
                "label": predicted_label,
                "predicted_confidence_score": predicted_confidence_score,
                "location": location,
                "center": center
            }, outfile)
            outfile.write('\n')

            if label == 1:
                print('Target found')
                lat, lon, alt, heading = location

                if ((vehicle.mode == AUTO or vehicle.mode == GUIDED) and (not found_target)):
                    descendAndReleaseImg(
                        vehicle, center[0], center[1], lat, lon, alt, heading)
                    vehicle.mode = AUTO
                    found_target = True
            elif label == 0:
                json_obj = {
                    "center": center,
                    "location": location,
                    "type": HOTSPOT
                }
                print("Hotspot found")

    im_array = r.plot()

    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
    cv2.imshow('YOLO Detection', im_array)

    out.write(im_array)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
