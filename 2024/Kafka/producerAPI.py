import cv2
from glob import glob
import concurrent.futures
import subprocess
from confluent_kafka import Producer
import os
import json
import time
import requests
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


class ProducerThread:
    def __init__(self, config):
        self.producer = Producer(config)

    def publishFrame(self, video_path):
        video = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        frame_no = 1
        while video.isOpened():
            _, frame = video.read()
            print(f"frame number done  === {frame_no}")
            frame_bytes = cv2.imencode(".jpeg", frame)[1].tobytes()
            print("frame recieved")
            result = subprocess.run(
                ["curl", "-s", "http://localhost:5000/location/"],
                capture_output=True,
                text=True
            )
            location = ""
            if result.returncode == 0:
                location = json.loads(result.stdout)
            print("location recieved is ",location)
            message_value = json.dumps({
                'frame': frame_bytes.decode('latin1'),
                'location': location
            }).encode('utf-8')
            # print(message_value)

            if frame_no % 1 == 0:
                self.producer.produce(
                    topic="single_video_stream",
                    value=message_value,
                    timestamp=frame_no,
                    headers={
                        "video_name": str.encode(video_name)
                    }
                )
                self.producer.poll(0.5)
            time.sleep(0.1)
            frame_no += 1
        video.release()
        return

    def start(self, vid_paths):
        self.publishFrame(video_path=vid_paths[0])
        self.producer.flush()
        print("Finished...")
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     executor.map(self.publishFrame, vid_paths)


if __name__ == "__main__":
    video_dir = "../../../../FineTuned_DERtModel/anotted.mp4"
    video_paths = glob(video_dir + "*.mp4")

    producer_thread = ProducerThread(producer_config)
    producer_thread.start([f"{video_dir}"])
