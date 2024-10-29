import socketio
import time
from ultralytics import YOLO
import json
import cv2
import threading
from drone_helper import connect_to_drone, goto_center, AUTO, GUIDED, Vehicle, align_at_center, drop_and_return_to_15
from classificationEnum import TARGET
from classificationEnum import HOTSPOT, TARGET, DET_OBJ
from drone_helper import connect_to_drone, getCurrentLocation
import time
from socketio.exceptions import TimeoutError


alignment_flag = [True, False, False, False]
alignment_state = 0
# 0, 1, 2

sio = socketio.SimpleClient()
# sio.connect('http://127.0.0.1:5000')


def emit_alignment(alignment_state, location, center):
    sio.emit("alignment", {"alignment_state": alignment_state,
             "location": location, "center": center})


# vehicle: Vehicle = connect_to_drone("tcp:localhost:5762")
# vehicle: Vehicle = connect_to_drone("udpout:10.42.0.1:10000")


MODEL_NAMES = ["./OptimizedWeights/best.pt"]
# SOURCES = ["../../../../videos/din.mp4"]
SOURCES = ["./output16.avi"]


def run_tracker_in_thread(model_name, filename):
    global alignment_state
    global alignment_flag
    frame_cnt = 0
    model = YOLO(model_name)
    results = model.track(filename, save=True,
                          stream=True, conf=0.70, iou=0.65)

    for r in results:
        # print(r)
        print(r.boxes.cls)
        print(r.boxes.conf)
        # time.sleep(0.5)


tracker_threads = []
for video_file, model_name in zip(SOURCES, MODEL_NAMES):
    thread = threading.Thread(target=run_tracker_in_thread, args=(
        model_name, video_file), daemon=True)
    tracker_threads.append(thread)
    thread.start()

for thread in tracker_threads:
    thread.join()

cv2.destroyAllWindows()
