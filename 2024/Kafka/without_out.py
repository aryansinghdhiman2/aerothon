# import numpy as np
# import cv2
# import os
# import random

# def adjust_gamma(image, gamma=1.0):
#     # Build a lookup table mapping pixel values [0, 255] to adjusted gamma values
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     # Apply gamma correction using the lookup table
#     return cv2.LUT(image, table)

# # Directory containing images
# input_dir = '../../../../LatestHotspots/Shapes/image/'

# # Adjust gamma for each image in the directory
# for image_file in os.listdir(input_dir):
#     image_path = os.path.join(input_dir, image_file)
#     # Load image
#     image = cv2.imread(image_path)
#     if image is not None:  # Check if image loaded correctly
#         # Generate a random gamma value in the range [1.1, 1.9]
#         random_float = random.uniform(1.1, 1.9)
#         print(f"Applying gamma {random_float} to {image_file}")
#         # Adjust image gamma
#         adjusted_image = adjust_gamma(image, gamma=random_float)
#         # Overwrite the original image with the adjusted image
#         cv2.imwrite(image_path, adjusted_image)
#     else:
#         print(f"Failed to load image {image_file}")
import cv2
import random
import os
from ultralytics import YOLO  

model = YOLO('./OptimizedWeights/best_5m.pt')
# results = model.track(".././/ShapeDetectionScripts/shapes.mp4", show=True)  # Tracking with default tracker
# print(results)


video_path = '../../../../test_yolo/videos/output2.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = './output_video_test_5.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break  
    height, width, _ = frame.shape

    # Define crosshair color and thickness
    color = (255, 0, 0) 
    thickness = 2


    results = model(frame, conf=0.75, iou=0.75)
    r = results[0]
    print(type(results),results)
    
    im_array = r.plot()
        # Draw horizontal line
    cv2.line(im_array, (width // 2 - 20, height // 2), (width // 2 + 20, height // 2), color, thickness)

    # Draw vertical line
    cv2.line(im_array, (width // 2, height // 2 - 20), (width // 2, height // 2 + 20), color, thickness)
    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
    cv2.imshow('YOLO Detection', im_array)

    out.write(im_array)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


