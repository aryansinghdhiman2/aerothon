from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def process_image(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    return outputs

def post_process(results, image, processor, threshold=0.9):
    target_sizes = torch.tensor([image.size[::-1]])
    processed_results = processor.post_process_object_detection(results, target_sizes=target_sizes, threshold=threshold)[0]
    return processed_results

def annotate_image(image, results, model, font=None):
    draw = ImageDraw.Draw(image)
    if font is None:
        font = ImageFont.load_default()
    
    box_coordinates = []
    for i in results['boxes']:
        for j in i:
            box_coordinates.append(float(j))
  
        
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=3)
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
        draw.text((box[0], box[1] - 10), label_text, fill="red", font=font)

    return (image,box_coordinates,results['labels'])

def save_annotated_image(image, save_path):
    image.save(save_path)

def main(image_path, processor, model, save_path):
    image = load_image(image_path)
    results = process_image(image, processor, model)
    processed_results = post_process(results, image, processor)
    annotated_results = annotate_image(image, processed_results, model)
    annotated_image=annotated_results[0]
    bbox_coordinates = annotated_results[1]
    label = annotated_results[2]
    save_annotated_image(annotated_image, save_path)
    annotated_image.show()

image_path = "./2024/img.jpeg"
save_path = "./2024/out.jpeg"


processor = AutoImageProcessor.from_pretrained("sansh2356/DETR_finetune")
model = AutoModelForObjectDetection.from_pretrained("sansh2356/DETR_finetune")

main(image_path, processor, model, save_path)
