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
sio.connect('http://127.0.0.1:5000')


def emit_alignment(alignment_state, location, center):
    sio.emit("alignment", {"alignment_state": alignment_state,
             "location": location, "center": center})


# vehicle: Vehicle = connect_to_drone("tcp:localhost:5762")
vehicle: Vehicle = connect_to_drone("udpout:10.42.0.1:10000")


MODEL_NAMES = ["./OptimizedWeights/best.pt"]
SOURCES = ["rtsp://localhost:8554/cam"]
# SOURCES = ["./output16.avi"]

def run_tracker_in_thread(model_name, filename):
    global alignment_state
    global alignment_flag
    frame_cnt = 0
    model = YOLO(model_name)
    results = model.track(filename, save=True,
                          stream=True, conf=0.70, iou=0.65)

    for r in results:
        locationObj = getCurrentLocation(vehicle)
        location = [locationObj.lat, locationObj.lon,
                    locationObj.alt, vehicle.heading]
        print(f"Frame count - {frame_cnt}")
        print("Location is - :::: ",location)
        frame_cnt+=1
        obj = (json.loads(r.to_json()))
        im_array = r.plot()
        cv2.namedWindow("Yolo detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Yolo detection", im_array)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
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

            for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
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
                    "label": predicted_label,
                    "predicted_confidence_score": predicted_confidence_score,
                    "center": center
                }, outfile)
                outfile.write('\n')

                if label == 1:
                    print('Target found')
                    # adjusting according to input resolution
                    adjusted_center = [center_x, (480 - center_y)]
                    print(f"Center {center}, adjusted_center {adjusted_center} location {location}")

                    # check alignment request state
                    if (alignment_state <= 2):
                        try:
                            event = sio.receive(timeout=0.5)
                        except TimeoutError:
                            pass
                        else:
                            alignment_state = event[1]
                            alignment_flag[alignment_state] = True
                            print('received event:', event[0], event[1])

                        if (vehicle.mode == AUTO or vehicle.mode == GUIDED):
                            if (alignment_flag[0]):
                                print("sent 0")
                                emit_alignment(alignment_state,
                                               location, adjusted_center)
                                alignment_flag[0] = False
                            elif (alignment_flag[1]):
                                print("sent 1")
                                emit_alignment(alignment_state,
                                               location, adjusted_center)
                                alignment_flag[1] = False
                            elif (alignment_flag[2]):
                                print("sent 2")
                                emit_alignment(alignment_state,
                                               location, adjusted_center)
                                alignment_flag[2] = False

                elif label == 0:
                    json_obj = {
                        "center": center,
                        "location": location,
                        "type": HOTSPOT
                    }
                    print("Hotspot found")

        print(type(results), results)
        pass


tracker_threads = []
for video_file, model_name in zip(SOURCES, MODEL_NAMES):
    thread = threading.Thread(target=run_tracker_in_thread, args=(
        model_name, video_file), daemon=True)
    tracker_threads.append(thread)
    thread.start()

for thread in tracker_threads:
    thread.join()

cv2.destroyAllWindows()
