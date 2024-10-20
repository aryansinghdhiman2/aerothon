from ultralytics import YOLO
import json
import cv2
import threading
from drone_helper import connect_to_drone, goto_center, AUTO, GUIDED, Vehicle, align_at_center, drop_and_return_to_15
from classificationEnum import TARGET
from classificationEnum import HOTSPOT, TARGET, DET_OBJ
from drone_helper import connect_to_drone, getCurrentLocation
import time


MODEL_NAMES = ["./OptimizedWeights/best.pt"]
SOURCES = ["../../../../NewVideos/output2.avi"]


def run_tracker_in_thread(model_name, filename):
    global found_target
    model = YOLO(model_name)
    results = model.track(filename, save=True, stream=True, conf=0.7, iou=0.65)

    for r in results:
        obj = (json.loads(r.to_json()))
        im_array = r.plot()
        cv2.namedWindow("YOlo detection", cv2.WINDOW_NORMAL)
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

                elif label == 0:
                    json_obj = {
                        "center": center,
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
