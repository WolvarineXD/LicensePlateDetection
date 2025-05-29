License Plate Detection & Recognition 🚗🔍
Overview
This project uses YOLO (v8 & v12) models to perform License Plate Detection and Recognition. It includes the training pipeline, model evaluation, inference on test images, and text extraction using Tesseract OCR. The goal of the project is to detect license plates from images and extract the text (plate number) for further processing.

Table of Contents
Project Setup

Training the Model

YOLOv8 Training

YOLOv12 Training

Model Evaluation

Inference and Text Extraction

Visualizing Training Metrics

Additional Notes

License

Project Setup
1. Prerequisites
Before running the scripts, ensure you have the following:

Python 3.8+ installed

Ultralytics YOLO: For training and inference (pip install ultralytics)

OpenCV: For image processing (pip install opencv-python)

Tesseract OCR: For text extraction from images (Tesseract Installation Guide)

Matplotlib: For plotting metrics (pip install matplotlib)

Pandas: For handling and visualizing metrics (pip install pandas)

2. Dataset
You should have a dataset of images with License Plates. The dataset should follow this directory structure:
data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/

You can use a dataset like Roboflow's License Plate Recognition Dataset. The data.yaml should look like:
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['License_Plate']

roboflow:
  workspace: roboflow-universe-projects
  project: license-plate-recognition-rxg4e
  version: 4
  license: CC BY 4.0
  url: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4

Training the Model
YOLOv8 Training
To train a YOLOv8 model, use the following script:
# train_yolov8.py
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="/content/license_plate_dataset/data.yaml",  # Path to data.yaml
    epochs=10,  # Number of epochs to train
    imgsz=640   # Image size
)
YOLOv12 Training
For YOLOv12, use the following script:
# train_yolov12.py
from ultralytics import YOLO

# Load the YOLOv12 model
model = YOLO("yolov12n.yaml")

# Train the model
model.train(
    data="/content/license_plate_dataset/data.yaml",  # Path to data.yaml
    epochs=10,  # Number of epochs to train
    imgsz=640   # Image size
)

import cv2
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
from glob import glob

# Load YOLOv8 license plate detection model
model = YOLO("/kaggle/working/runs/detect/yolov12_license_plate/weights/best.pt")

# Set path to folder containing images/kaggle/input/dataset-5/License Plate Recognition.v4-resized640_aug3x-accurate.yolov8/test/images
image_folder = "/kaggle/input/dataset-5/License Plate Recognition.v4-resized640_aug3x-accurate.yolov8/test/images"

# Get list of image files
image_paths = glob(os.path.join(image_folder, ".jpg"))  # you can add ".png" if needed

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Process each image in the folder
for image_path in image_paths:
    print(f"\n📷 Processing: {os.path.basename(image_path)}")
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))

    results = model(image)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Add padding
            pad = 10
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)

            # Crop plate region
            plate_image = image[y1:y2, x1:x2]

            # Show cropped plate
            plt.figure(figsize=(6, 3))
            plt.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
            plt.title("Detected License Plate")
            plt.axis("off")
            plt.show()

            # OCR
            ocr_result = reader.readtext(plate_image)

            if ocr_result:
                plate_text = ' '.join([res[1] for res in ocr_result])
                print("🔍 Detected License Plate Number:", plate_text)

                # Draw on original image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, plate_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                print("❌ No text detected")

    # Show final result
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Final Result: {os.path.basename(image_path)}")
    plt.axis("off")
    plt.show()
    # visualize_metrics.py
import pandas as pd
import matplotlib.pyplot as plt

# Load the results from the training process
df = pd.read_csv("/content/runs/detect/yolov12_license_plate/results.csv")

# Plot mAP, Precision, Recall, and Loss metrics
plt.figure(figsize=(12, 8))

# Plot mAP at 0.5
plt.plot(df["metrics/mAP50(B)"], label="mAP@0.5")
plt.plot(df["metrics/precision(B)"], label="Precision")
plt.plot(df["metrics/recall(B)"], label="Recall")
plt.plot(df["train/box_loss"], label="Box Loss")

plt.title("Training Metrics for YOLOv12")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)
plt.show()

Additional Notes
Training Duration: The number of epochs and training time depends on the dataset size and hardware. Typically, 50-100 epochs are recommended for better results.

OCR Accuracy: Tesseract OCR's accuracy is highly dependent on image quality, preprocessing steps, and the clarity of the license plate numbers.

Post-processing: Additional processing like non-maximum suppression or custom plate validation can improve detection results.

License
This project uses datasets from Roboflow and is licensed under CC BY 4.0
End of README
This README file provides a complete guide to setting up, training, evaluating, and testing your License Plate Detection & Recognition system. Let me know if you'd like to add or adjust anything else!