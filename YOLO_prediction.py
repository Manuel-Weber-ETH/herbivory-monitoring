# cd "C:\0_Documents\10_ETH\Thesis\Python"
# env\Scripts\activate

root_directory = "C:/0_Documents/10_ETH/Thesis_old/test"

import os
import pandas as pd
from ultralytics import YOLO
from PIL import Image, ExifTags
from datetime import datetime
import argparse
import time

def check_and_remove_corrupted_files(directory_path):
    count = 0
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except (IOError, SyntaxError) as e:
                    print(f"Corrupted file detected and removed: {file_path}")
                    os.remove(file_path)
                    count += 1
    print(f"Total corrupted files removed: {count}")

def get_image_metadata(image_path):
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data is not None:
                for tag, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    if tag_name == "DateTimeOriginal":
                        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception as e:
        print(f"Error extracting metadata: {e}")
    return None

def process_folder(folder_path, model):
    check_and_remove_corrupted_files(folder_path)
    
    results = model.predict(source=os.path.join(folder_path, "*/*"),
                            stream=True,
                            save=False,
                            conf=0.75)
    
    species_names = list(model.names.values())
    
    # Initialize DataFrame with additional columns for individual counts
    columns = ['Name', 'Date_Time'] + [f'{species}_confidence' for species in species_names] + [f'{species}_count' for species in species_names]
    df = pd.DataFrame(columns=columns)

    for result in results:
        date_time = get_image_metadata(result.path) or "NA"
        
        # Initialize dictionaries for confidence scores and counts
        species_confidences = {f'{species}_confidence': 0 for species in species_names}
        species_counts = {f'{species}_count': 0 for species in species_names}
        
        if result.boxes is not None and len(result.boxes) > 0:  # Check if there are any boxes detected
            conf = result.boxes.conf.numpy()  # Confidence scores
            classes = result.boxes.cls.numpy()  # Class IDs
            for idx, cls in enumerate(classes):
                species = model.names[int(cls)]
                if conf[idx] > species_confidences[f'{species}_confidence']:
                    species_confidences[f'{species}_confidence'] = conf[idx]
                species_counts[f'{species}_count'] += 1
        
        # Create row for the current image
        row = [result.path, date_time] + [species_confidences[f'{species}_confidence'] for species in species_names] + [species_counts[f'{species}_count'] for species in species_names]
        df.loc[len(df)] = row
    
    csv_file = os.path.join(os.path.dirname(folder_path), f"{os.path.basename(folder_path)}_full_image_annotations.csv")
    df.to_csv(csv_file, index=False)
    print(f"CSV file saved: {csv_file}")

def main(root_directory):
    start_time = time.time()
    model = YOLO("yoloweightsv10_lib.pt")
    # Only process folders directly in the root directory
    for item in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, item)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path}")
            process_folder(folder_path, model)
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main(root_directory)

'''
# YOLO validation

### Validation on test split YOLOv8
from ultralytics import YOLO

# Load a model
model = YOLO("yoloweights.pt")  # load a custom model

# Validate the model
metrics = model.val(data = 'valid.yaml', plots = True, conf=0.75 )  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
'''
'''
### Validation on test split YOLOv10
from ultralytics import YOLOv10

model = YOLOv10('YOLOv10weights/epoch40.pt')

#model = YOLOv10.from_pretrained('YOLOv10weights/yolov10l')

metrics = model.val(data='valid.yaml', batch=16, conf=0.75)
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category


# Speed test yolo

from ultralytics import YOLO
model = YOLO("yolov10s.yaml")
outs = model.predict(source="0", show=False)


from ultralytics import YOLO
import os

# Load the model
#model = YOLO("best.pt")
model = YOLO("yoloweightsv8.pt")
print("Model loaded")

# Define the image path
image_path = "C:/0_Documents/10_ETH/Thesis/test/ongava 2021/ADC"

# Perform object detection on images within the directory
results = model.predict(source=os.path.join(image_path), save=False, show=False, conf=0.75, stream=True)

# Process results if needed
for result in results:
    print(result)
    
'''