# # from ultralytics.models.yolo import YOLO
# # from ultralytics import YOLO
# # import json
# # import cv2
# # import threading
# # from drone_helper import connect_to_drone, goto_center, AUTO, GUIDED, Vehicle, align_at_center, drop_and_return_to_15
# # from classificationEnum import TARGET
# # from classificationEnum import HOTSPOT, TARGET, DET_OBJ
# # from drone_helper import connect_to_drone, getCurrentLocation
# # import time


# # MODEL_NAMES = ["./OptimizedWeights/best.pt"]
# # SOURCES = ["rtsp://localhost:8554/cam"]


# # def run_tracker_in_thread(model_name, filename):
# #     global found_target
# #     model = YOLO(model_name)
# #     results = model.track(filename, save=True, stream=True, conf=0.7, iou=0.65)

# #     for r in results:
# #         obj = (json.loads(r.to_json()))
# #         im_array = r.plot()
# #         cv2.namedWindow("YOlo detection", cv2.WINDOW_NORMAL)
# #         cv2.imshow("Yolo detection", im_array)

# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break
# #         conv_obj = {}
# #         if len(obj) > 0:
# #             conv_obj = {
# #                 "boxes": [obj[0]['box']],
# #                 "labels": [obj[0]['class']],
# #                 "scores": [obj[0]['confidence']]
# #             }
# #         else:
# #             conv_obj = {
# #                 "boxes": [],
# #                 "labels": "none",
# #                 "scores": []
# #             }
# #         with open("./json_output.jsonl", "a") as outfile:
# #             res = conv_obj
# #             if len(res["scores"]) == 0:
# #                 json.dump({
# #                     "box": "None",
# #                     "label": "None",
# #                     "predicted_confidence_score": "None",
# #                     "location": []
# #                 }, outfile)
# #                 outfile.write('\n')

# #             for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
# #                 box = (box['x1'], box['y1'],
# #                        box['x2'], box['y2'])

# #                 x_min, y_min, x_max, y_max = box
# #                 center_x = (x_min + x_max) / 2
# #                 center_y = (y_min + y_max) / 2

# #                 center = (center_x, center_y)

# #                 predicted_label = label
# #                 predicted_confidence_score = score
# #                 json.dump({
# #                     "box": box,
# #                     "label": predicted_label,
# #                     "predicted_confidence_score": predicted_confidence_score,
# #                     "center": center
# #                 }, outfile)
# #                 outfile.write('\n')
# #                 if label == 1:
# #                     print("CENTER", center)
# #                     print('Target found')

# #                 elif label == 0:
# #                     json_obj = {
# #                         "center": center,
# #                         "type": HOTSPOT
# #                     }
# #                     print("Hotspot found")

# #         print(type(results), results)
# #         pass


# # tracker_threads = []
# # for video_file, model_name in zip(SOURCES, MODEL_NAMES):
# #     thread = threading.Thread(target=run_tracker_in_thread, args=(
# #         model_name, video_file), daemon=True)
# #     tracker_threads.append(thread)
# #     thread.start()

# # for thread in tracker_threads:
# #     thread.join()

# # cv2.destroyAllWindows()
# # import cv2
# # import math
# from ultralytics.models.yolo import YOLO
# import torch

# # # Open the default camera
# # cam = cv2.VideoCapture(0)

# # # Get the default frame width and height
# # frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
# # frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # # Define the codec and create VideoWriter object
# # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
# model = YOLO('./OptimizedWeights/best.pt')
# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# results = model.track(source="rtsp://localhost:8554/cam",show=True)
# # model = model.to(device=device)
# # while True:
# #     ret, frame = cam.read()
# #     results = model(frame, stream=True)

# #     # Write the frame to the output file
# #     out.write(frame)
# #     for r in results:
# #         boxes = r.boxes
# #         for box in boxes:
# #             # bounding box
# #             x1, y1, x2, y2 = box.xyxy[0]
# #             x1, y1, x2, y2 = int(x1), int(y1), int(
# #                 x2), int(y2)  # convert to int values

# #             # put box in cam
# #             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

# #             # confidence
# #             confidence = math.ceil((box.conf[0]*100))/100
# #             print("Confidence --->", confidence)

# #             # class name
# #             cls = int(box.cls[0])
# #             # print("Class name -->", classNames[cls])

# #             # object details
# #             org = [x1, y1]
# #             font = cv2.FONT_HERSHEY_SIMPLEX
# #             fontScale = 1
# #             color = (255, 0, 0)
# #             thickness = 2

# #             cv2.putText(frame,"label", org,
# #                         font, fontScale, color, thickness)

# #     cv2.imshow('Webcam', frame)
# #     if cv2.waitKey(0) == ord('q'):
# #         break

# # # Release the capture and writer objects
# # cam.release()
# # out.release()
# # cv2.destroyAllWindows()


import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer
from aiohttp import web

# Initialize peer connections dictionary
pcs = set()

# WebRTC endpoint handler


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Set up the peer connection
    pc = RTCPeerConnection()
    pcs.add(pc)

    # Handle video stream
    player = MediaPlayer("ice://localhost:8189/cam")
    pc.addTrack(player.video)

    # Handle the incoming offer and create an answer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Send answer back to the client
    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )

# Clean up peer connections


async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

# Run server
app = web.Application()
app.on_shutdown.append(on_shutdown)
app.router.add_post("/offer", offer)

web.run_app(app, port=8888)
