import cv2
import random
import torch
import os

from ultralytics import YOLO

model = YOLO('../YoloNew/best.pt')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(device)

video_path = 'rtsp://10.42.0.1:8554/cam'

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
    if not ret:
        break

    results = model(frame, conf=0.5, iou=0.6)
    r = results[0]
    print(type(results), results)

    im_array = r.plot()

    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
    cv2.imshow('YOLO Detection', im_array)

    out.write(im_array)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
