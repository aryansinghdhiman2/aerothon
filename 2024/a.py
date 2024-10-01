import glob
from roboflow import Roboflow
import os

# Initialize Roboflow client
rf = Roboflow(api_key="4PZ7anto2pXwMrpLvG7e")

# Directory path and file extension for images
dir_name = "./NewDataset/Final/images/"
file_extension_type = ".png"

# Annotation file path and format (e.g., .coco.json)
# Get the upload project from Roboflow workspace
project = rf.workspace().project("detr-dataset-kvywo")

# Upload images
image_glob = glob.glob(dir_name + '/*' + file_extension_type)
for image_path in image_glob:
    image_number = os.path.splitext(os.path.basename(image_path))[0]
    annotation_filename = f"./NewDataset/Final/labels/{image_number}.txt"
    print(project.single_upload(
        image_path=image_path,
        annotation_path=annotation_filename,
        # optional parameters:
        # annotation_labelmap=labelmap_path,
        # split='train',
        # num_retry_uploads=0,
        # batch_name='batch_name',
        # tag_names=['tag1', 'tag2'],
        # is_prediction=False,
    ))
