# yolov8_detection.py

from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model
model = YOLO('best.pt')  # Replace with your .pt file path

def detect_objects(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Run detection
    results = model(image)

    # Annotate image with bounding boxes and labels
    annotated_image = results[0].plot()  # This will return an annotated image
    
    return annotated_image  # Return the annotated image
