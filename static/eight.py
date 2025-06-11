# python seven.py


import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import logging
import sys

# Set up logging to both console and file
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "seven.log")

# Create file handler
file_handler = logging.FileHandler(log_file, mode='a')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# Set up the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def seven(input_path):
    """
    Run YOLO object detection on an input image and return the annotated image 
    with only the highest confidence detection box and detected text.
    """
    try:
        input_path = input_path.strip()  # Clean trailing spaces
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {input_path}")
        logging.info(f"Image loaded from {input_path}")
        logging.info(f"Image dimensions: {img.shape[1]}x{img.shape[0]}")

        original_img = img.copy()

        # Load the YOLO model
        try:
            model_path = r"D:\\rahul_jangam\\FJP_LIVE_DEMO\\FJP\\best.pt"
            model = YOLO(model_path)
            logging.info(f"YOLO model loaded from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {str(e)}")
            return None, None

        # Run inference with specific confidence and IoU thresholds
        try:
            results = model(original_img, conf=0.8, iou=0.8)[0]
            if results is None or len(results.boxes) == 0:
                logging.warning(f"No objects detected in image: {input_path}")
                return original_img, []
            
            logging.info(f"YOLO detection found {len(results.boxes)} objects")
        except Exception as e:
            logging.error(f"YOLO detection failed: {str(e)}")
            return None, None

        annotated_img = original_img.copy()
        detected_text = []

        # Process only the highest confidence detection
        try:
            if results.boxes and len(results.boxes) > 0:
                # Find the box with the highest confidence
                best_box = max(results.boxes, key=lambda b: float(b.conf[0]))
                
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                conf = float(best_box.conf[0])
                cls_id = int(best_box.cls[0])
                label = f"{model.names[cls_id]} {conf:.2f}"
                
                logging.info(f"Best detection: {label} at position ({x1},{y1})-({x2},{y2})")
                
                # Add to detected text list
                detected_text.append(label)
                
                # Draw the highest confidence box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_img, 
                    label, 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
        except Exception as e:
            logging.error(f"Error processing detection results: {str(e)}")
            return None, None

        # Save to output directory
        try:
            output_dir = r"D:\\rahul_jangam\\FJP_LIVE_DEMO\\code\\output\\final_output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Get original filename without path
            base_filename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, base_filename)
            
            if not cv2.imwrite(output_path, annotated_img):
                raise IOError(f"Failed to save output image to: {output_path}")
            logging.info(f"Annotated image saved to: {output_path}")
        except Exception as e:
            logging.warning(f"Error saving output image: {str(e)}")

        # Save backup image
        try:
            backup_dir = r"D:\\rahul_jangam\\FJP_LIVE_DEMO\\code\\output_backup"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Add timestamp to backup image
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"detected_{current_datetime}.jpg"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            if not cv2.imwrite(backup_path, annotated_img):
                raise IOError(f"Failed to save backup image to: {backup_path}")
            logging.info(f"Backup image saved to: {backup_path}")
        except Exception as e:
            logging.warning(f"Error saving backup image: {str(e)}")

        return annotated_img, detected_text

    except Exception as e:
        logging.error(f"Unexpected error in seven function: {str(e)}")
        return None, None