# import threading
# from confluent_kafka import Consumer, KafkaError, KafkaException
# import cv2
# import numpy as np
# import json
# from PIL import Image
# import time
# import torch
# from transformers import AutoImageProcessor, AutoModelForObjectDetection

# consumer_config = {
#     'bootstrap.servers': '127.0.0.1:9092',
#     'group.id': 'single_video_stream',
#     'enable.auto.commit': False,
#     'default.topic.config': {'auto.offset.reset': 'earliest'}
# }


# def draw_boxes(self, image, results):
#     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#         box = [round(i, 2) for i in box.tolist()]
#         predicted_label = label.item()
#         predicted_confidence_score = score.item()

#         start_point = (int(box[0]), int(box[1]))
#         end_point = (int(box[2]), int(box[3]))
#         color = (0, 255, 0)

#         thickness = 2
#         cv2.rectangle(image, start_point, end_point, color, thickness)

#         label_text = f"""Label: {predicted_label}, Score: {
#             predicted_confidence_score:.2f}"""
#         cv2.putText(image, label_text, start_point,
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

#     return image


# def initialize_video_writer(self, frame_size, output_path="output_video.mp4", fps=30):
#     self.frame_width, self.frame_height = frame_size
#     self.video_writer = cv2.VideoWriter(
#         output_path,
#         cv2.VideoWriter_fourcc(*'mp4v'),
#         fps,
#         (self.frame_width, self.frame_height)
#     )


# class ConsumerThread:
#     def __init__(self, config, topic, batch_size, model, processor,device):
#         self.config = config
#         self.topic = topic
#         self.batch_size = batch_size
#         self.model: AutoModelForObjectDetection = model
#         self.processor = processor
#         self.device = device

#     def read_data(self):
#         consumer = Consumer(self.config)
#         print(type(self.topic))
#         consumer.subscribe(self.topic)
#         self.run(consumer, 0, [], [])

#     def run(self, consumer, msg_count, msg_array, metadata_array):
#         try:
#             while True:
#                 msg = consumer.poll(0)
#                 print(msg)
#                 if msg == None:
#                     continue
#                 elif msg.error() == None:

#                     # convert image bytes data to numpy array of dtype uint8
#                     nparr = np.frombuffer(msg.value(), np.uint8)

#                     # decode image
#                     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#                     msg_array.append(img)

#                     # get metadata
#                     frame_no = msg.timestamp()[1]
#                     video_name = msg.headers()[0][1].decode("utf-8")
#                     print(f"Frame number {frame_no}")
#                     metadata_array.append((frame_no, video_name))
#                     # bulk process
#                     msg_count += 1
#                     if msg_count % self.batch_size == 0:
#                         for frame in msg_array:

#                             image = Image.fromarray(
#                                 cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                             cv2.waitKey(0)
#                             cv2.destroyAllWindows()
#                             inputs = processor(
#                                 images=image, return_tensors="pt").to(device)
#                             outputs = model(**inputs)
#                             target_sizes = torch.tensor([image.size[::-1]])

#                             results = processor.post_process_object_detection(
#                                 outputs, target_sizes=target_sizes, threshold=0.9)[0]
#                             # print(results['scores'],results['labels'],results["boxes"])
#                             with open("./json_output.jsonl", "a") as outfile:
#                                 if (len(results["scores"]) == 0):
#                                     json.dump({
#                                         "box": "None",
#                                         "frame_no": frame_no,
#                                         "video_name": video_name,
#                                         "label": "None",
#                                         "predicted_confidence_score": "None"
#                                     }, outfile)
#                                     outfile.write('\n')

#                                 for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#                                     print(score, label, box)
#                                     box = [round(i, 2) for i in box.tolist()]
#                                     predicted_label = label.item()
#                                     predicted_confidence_score = score.item()
#                                     print(box, predicted_confidence_score,
#                                           predicted_label)
#                                     json.dump({
#                                         "box": box,
#                                         "frame_no": frame_no,
#                                         "video_name": video_name,
#                                         "label": predicted_label,
#                                         "predicted_confidence_score": predicted_confidence_score
#                                     }, outfile)
#                                     outfile.write('\n')
#                             annotated_img = draw_boxes(self, frame, results)

#                             consumer.commit(asynchronous=False)
#                             msg_count = 0
#                             metadata_array = []
#                             msg_array = []

#                 elif msg.error().code() == KafkaError._PARTITION_EOF:
#                     print('End of partition reached {0}/{1}'
#                           .format(msg.topic(), msg.partition()))
#                 else:
#                     print('Error occured: {0}'.format(msg.error().str()))

#         except KeyboardInterrupt:
#             print("Detected Keyboard Interrupt. Quitting...")
#             pass

#         finally:
#             consumer.close()

#     def start(self, numThreads):
#         for _ in range(numThreads):
#             t = threading.Thread(target=self.read_data)
#             t.daemon = True
#             t.start()
#             while True:
#                 time.sleep(10)


# if __name__ == "__main__":
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     topic = ["single_video_stream"]
#     processor = AutoImageProcessor.from_pretrained("sansh2356/DETR_finetune")
#     model = AutoModelForObjectDetection.from_pretrained(
#         "sansh2356/DETR_finetune")
#     model = model.to(device)
#     print(device)

#     print("model and processor loaded")
#     time.sleep(5)

#     consumer_thread = ConsumerThread(
#         consumer_config, topic, 1, model, processor,device)
#     consumer_thread.start(3)


import threading
from confluent_kafka import Consumer, KafkaError
import cv2
import json
import time
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from drone_helper import connect_to_drone,descendAndReleaseImg,LocationGlobal,get_distance_metres,AUTO

from classificationEnum import TARGET

vehicle = connect_to_drone("tcp:localhost:5762")

hotspots: list[LocationGlobal] = []

consumer_config = {
    'bootstrap.servers': '127.0.0.1:9092',
    'group.id': 'single_video_stream',
    'enable.auto.commit': False,
    'default.topic.config': {'auto.offset.reset': 'earliest'}
}


class ConsumerThread:
    def __init__(self, config, topic, batch_size):
        self.config = config
        self.topic = topic
        self.batch_size = batch_size

    def read_data(self):
        print('read_data called')
        consumer = Consumer(self.config)
        consumer.subscribe(self.topic)
        self.run(consumer, 0, [], [])

    def run(self, consumer, msg_count, msg_array, metadata_array):
        try:
            while True:
                msg = consumer.poll(0)
                print(msg)
                time.sleep(3)
                if msg is None:
                    continue

                elif msg.error() is None:
                    print('Message received successfully')
                    bytes_arr = msg.value().decode("utf-8")
                    json_obj = json.loads(bytes_arr)
                    center = json_obj["center"]
                    print(center)
                    frame_no = msg.timestamp()[1]
                    video_name = msg.headers()[0][1].decode("utf-8")
                    print(f"Frame number {frame_no}")
                    metadata_array.append((frame_no, video_name))

                    msg_count += 1

                    consumer.commit(asynchronous=False)
                    msg_count = 0
                    metadata_array = []

                    lat, lon, alt, heading = json_obj["location"]
                    found_matching = False
                    if(json_obj['type'] == TARGET):
                        for hotspot in hotspots:
                            if(get_distance_metres(hotspot,LocationGlobal(lat,lon)) < 5):
                                found_matching = True
                                break
                        
                        if(not found_matching):
                            descendAndReleaseImg(vehicle,center[0],center[1],lat,lon,alt,heading,hotspots)
                            vehicle.mode = AUTO
                        

                elif msg.error().code() == KafkaError._PARTITION_EOF:
                    print('End of partition reached {0}/{1}'
                          .format(msg.topic(), msg.partition()))
                else:
                    print('Error occurred: {0}'.format(msg.error().str()))

        except KeyboardInterrupt:
            print("Detected Keyboard Interrupt. Quitting...")
            pass

        finally:
            consumer.close()
            cv2.destroyAllWindows()

    def start(self, numThreads):
        for _ in range(numThreads):
            t = threading.Thread(target=self.read_data)
            t.daemon = True
            t.start()
            while True:
                time.sleep(10)


if __name__ == "__main__":
    consumer_thread = ConsumerThread(
        consumer_config, ["single_video_stream"], 1)
    consumer_thread.start(3)
