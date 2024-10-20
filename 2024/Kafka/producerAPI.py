import cv2
from glob import glob
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import subprocess
from confluent_kafka import Producer
import os
from PIL import Image
import json
import time
import torch
from classificationEnum import HOTSPOT,TARGET,DET_OBJ
from drone_helper import connect_to_drone,getCurrentLocation

# vehicle = connect_to_drone("udpout:10.42.0.1:11000")
vehicle = connect_to_drone("tcp:localhost:5762")

message_value:bytes = ''
# Basic configuration for kafka producer

producer_config = {
    'bootstrap.servers': 'localhost:9092',
    'enable.idempotence': True,
    'acks': 'all',
    'retries': 100,
    'max.in.flight.requests.per.connection': 5,
    'compression.type': 'snappy',
    'linger.ms': 5,
    'batch.num.messages': 1
}


def draw_boxes(image, results):
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        predicted_label = label.item()
        predicted_confidence_score = score.item()

        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        color = (0, 255, 0)
        center_point = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        print("Center of the bbox", center_point)
        thickness = 2
        cv2.rectangle(image, start_point, end_point, color, thickness)

        label_text = f"""Label: {predicted_label}, Score: {
            predicted_confidence_score:.2f}"""
        cv2.putText(image, label_text, start_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    return image


class ProducerThread:
    def __init__(self, config, model, processor, device):
        self.producer = Producer(config)
        self.model = model
        self.device = device
        self.processor = processor

    def publishAndDetectFrame(self, video_path):
        global message_value

        video = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        frame_no = 1
        while video.isOpened():
            locationObj = getCurrentLocation(vehicle)
            location = [locationObj.lat,locationObj.lon,locationObj.alt,vehicle.heading]

            _, frame = video.read()
            print(f"frame number done  === {frame_no}")
            image = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.processor(
                images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])

            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.9)[0]
            print(results)

            center = (0,0)
            with open("./json_output.jsonl", "a") as outfile:
                if len(results["scores"]) == 0:
                    json.dump({
                        "box": "None",
                        "frame_no": frame_no,
                        "video_name": video_name,
                        "label": "None",
                        "predicted_confidence_score": "None",
                        "location": location
                    }, outfile)
                    outfile.write('\n')

                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    ### ---------------- All Detections happen here -------------------------------- ###
                    box = [round(i, 2) for i in box.tolist()]
                    # print(f"score: {type(score)} label: {type(label)}, box : {type(box)}")
                    print(box)
                    x_min, y_min, x_max, y_max = box

                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2

                    center = (center_x, center_y)

                    predicted_label = label.item()
                    predicted_confidence_score = score.item()
                    json.dump({
                        "box": box,
                        "frame_no": frame_no,
                        "video_name": video_name,
                        "label": predicted_label,
                        "predicted_confidence_score": predicted_confidence_score,
                        "location": location

                    }, outfile)
                    outfile.write('\n')

                    message_value = json.dumps({
                        'center': center,
                        'location': location,
                        'type' : TARGET
                    }).encode('utf-8')

                    self.producer.produce(
                        topic="single_video_stream",
                        value=message_value,
                        timestamp=frame_no,
                        headers={
                            "video_name": str.encode(video_name)
                        }
                    )
                    
                    annotated_img = draw_boxes(frame, results)

                    cv2.imshow("Processed Frame", annotated_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            print(results)


            self.producer.poll(0.5)
            time.sleep(0.1)
            frame_no += 1
        video.release()
        return

    def start(self, vid_paths):
        print("model and processor loaded")
        self.publishAndDetectFrame(video_path=vid_paths[0])
        self.producer.flush()
        print("Finished...")
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     executor.map(self.publishFrame, vid_paths)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    topic = ["single_video_stream"]
    processor = AutoImageProcessor.from_pretrained(
        "sansh2356/DETR_finetune")
    model = AutoModelForObjectDetection.from_pretrained(
        "sansh2356/DETR_finetune")
    model = model.to(device)
    print(device)
    video_dir = "./Kafka/"
    video_paths = glob(video_dir + "*.mp4")

    producer_thread = ProducerThread(
        producer_config, model, processor, device=device)
    producer_thread.start([f"{video_dir}"])
