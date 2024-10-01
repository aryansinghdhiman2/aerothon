# %%
from transformers import AutoImageProcessor, AutoModelForObjectDetection

processor = AutoImageProcessor.from_pretrained("sansh2356/DETR_finetune")
model = AutoModelForObjectDetection.from_pretrained("sansh2356/DETR_finetune")

# %%
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont

image_path = "./NewDataset/images/00017797.png"
image = Image.open(image_path).convert("RGB")


inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

draw = ImageDraw.Draw(image)
font = ImageFont.load_default()  

for i in results["boxes"]:
  print(i)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    
    draw.rectangle(box, outline="red", width=3)
    
    label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
    
    draw.text((box[0], box[1] - 10), label_text, fill="red", font=font)

image.show()  
image.save("./test/annotated_image.png") 


# %%
import os 
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import time 

video_paths = os.listdir('./NewVideos/')
def process_video(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline="red", width=3)

            label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
            draw.text((box[0], box[1] - 10), label_text, fill="red", font=font)

        frame_with_detections = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        print(f"frame done with count {frame_count}")
        out.write(frame_with_detections)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Video saved as {output_video_path}")
cnt = 1
for video_path in video_paths:
    complete_path = os.path.join("./NewVideos/",video_path)
    output_video_path =f"./FineTuned_DERtModel/anotted_video{5}.mp4"
    curr_t = time.time()
    process_video(complete_path, output_video_path)
    end_t = time.time()
    print(f"{video_path} has been processed in time taken : === {end_t-curr_t}")
    cnt+=1




