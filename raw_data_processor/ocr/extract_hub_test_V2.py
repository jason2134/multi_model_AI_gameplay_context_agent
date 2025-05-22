import easyocr
import cv2
import numpy as np
import os
import csv
import glob
import random
import pandas as pd
from natsort import natsorted

class OCR_DataProcessor:
    def __init__(self, input_path, output_folder):
        self.input_path = input_path
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
    def crop_image(self,image, frame_file, averaged_roi, param): # Function for cropping a sinlg ROI from the same image
        if image is None:
            print(f"Error: Could not load image at {frame_file}")
            return None
        # Get the dimensions of the image
        height, width = image.shape[:2]
        print(f"\nProcessing {os.path.basename(frame_file)} - Dimensions: {width}x{height}")

        x_center, y_center, roi_width, roi_height = averaged_roi
        # Convert normalized coordinates to pixel values
        x_center_pixel = int(x_center * width) #correct 254
        y_center_pixel = int(y_center * height) # correct 453
        roi_width_pixel = int(roi_width * width) # correct 76
        roi_height_pixel = int(roi_height * height) # correct 52
            
        # Calculate the top-left and bottom-right corners of the bounding box
        x1 = max(0, x_center_pixel - roi_width_pixel // 2) # 216
        y1 = max(0, y_center_pixel - roi_height_pixel // 2)
        x2 = min(width, x_center_pixel + roi_width_pixel // 2) #  292 
        y2 = min(height, y_center_pixel + roi_height_pixel // 2) # 480

        # Slightly expand the ROI
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        print(f"Adjusted ROI:(x1={x1}, y1={y1}, x2={x2}, y2={y2})")
            
        # Crop the image to the ROI
        cropped_image = image[y1:y2, x1:x2]
        # Save the cropped image
        cropped_image_path = os.path.join(f'{self.output_folder}/{param}', f"{os.path.basename(frame_file)}")
        
        # Generate the new filename with the parameter suffix (e.g., "_ammo")
        base_filename = os.path.basename(frame_file)
        new_filename = base_filename.replace(".jpg", f"_{param}.jpg")  # Add suffix before extension
        cropped_image_path = os.path.join(f'{self.output_folder}/{param}', new_filename)  
        cropped_image_name = f'{os.path.basename(cropped_image_path)}'
        
        cv2.imwrite(cropped_image_path, cropped_image)
        return cropped_image, cropped_image_name
            
    def ocr_label_csv_generation(self, chunk_size = 50):
        reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a compatible GPU
        csv_path = os.path.join(self.output_folder, "label.csv")
        #csv_header = ['file_name', 'predicted_results', 'confidence']
        csv_header = ['file_name', 'pred_roi_ammo', 'conf_roi_ammo', 
                      'pred_roi_health', 'conf_roi_health', 
                      'pred_roi_armour', 'conf_roi_armour', 
                      'roi_ammo_file_name', 'roi_health_file_name', 'roi_armour_file_name']
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=csv_header).to_csv(csv_path, index=False)
            
        data = []
        frame_files = natsorted(sorted([os.path.join(self.input_path, f) for f in os.listdir(self.input_path) if f.endswith(('.png', '.jpg', '.jpeg'))]))
        try:
            for frame_file in frame_files:
                image = cv2.imread(frame_file)
                row = {
                }
                row['file_name'] = os.path.basename(frame_file) # Add the file_name
                averaged_roi = {
                            'ammo': (0.184150,0.945098, 0.122549, 0.109804),
                            'health': (0.398366,0.948366,0.121569,0.103268),
                            'armour': (0.616993, 0.945098, 0.117647, 0.109804)
                }
                for param in averaged_roi.keys():
                    cropped_image, cropped_image_name = self.crop_image(image, frame_file, averaged_roi[param], param)
                    row[f'roi_{param}_file_name'] = cropped_image_name
                    # preprocess the cropped image
                    cropped_image = cv2.convertScaleAbs(cropped_image, alpha=2.0, beta=0) # beta = 10
                    results = reader.readtext(cropped_image, detail=1, paragraph=False,  allowlist='0123456789')
                    
                    if not results:
                        cropped_image = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                        results = reader.readtext(cropped_image, detail=1, paragraph=False, allowlist='0123456789')
                        if not results:
                            cropped_image = cv2.resize(cropped_image, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
                            results = reader.readtext(cropped_image, detail=1, paragraph=False, allowlist='0123456789')
            
                    if results:     
                        for (bbox, text, prob) in results:
                            row[f'pred_roi_{param}'] = text
                            row[f'conf_roi_{param}']  = prob
                            #data.append([cropped_image_name, text, prob])
                    else:
                        #data.append([cropped_image_name, 'No text detected', 0.0])
                        row[f'pred_roi_{param}'] = 'No text detected'
                        row[f'conf_roi_{param}']  = 0.0
                        
                data.append([row['file_name'], row['pred_roi_ammo'], row['conf_roi_ammo'],
                             row['pred_roi_health'], row['conf_roi_health'], 
                             row['pred_roi_armour'], row['conf_roi_armour'],
                             row['roi_ammo_file_name'], row['roi_health_file_name'], row['roi_armour_file_name']])
                
                row.clear()
                del image
                del cropped_image
                del cropped_image_name
                del results
                
                if len(data) >= chunk_size:
                    try:
                        self.save_to_csv(data, csv_path)
                        data.clear()
                        print(f"Saved chunk of {chunk_size} frames to {csv_path}")
                    except Exception as e:
                        print(f"Error saving to CSV: {e}")
            if data:
                try:
                    self.save_to_csv(data, csv_path)
                    print(f"Saved final {len(data)} frames to {csv_path}")
                except Exception as e:
                    print(f"Error saving final data to CSV: {e}")
            print(f'Saved total {len(frame_files)} to CSV')
        except:
            return None
        
    def save_to_csv(self, data, output_csv):
        csv_header = ['file_name', 'pred_roi_ammo', 'conf_roi_ammo', 
                      'pred_roi_health', 'conf_roi_health', 
                      'pred_roi_armour', 'conf_roi_armour', 
                      'roi_ammo_file_name', 'roi_health_file_name', 'roi_armour_file_name']
        df = pd.DataFrame(data, columns=csv_header)
        df.to_csv(output_csv, mode='a', header=False, index=False)  # Append without headers each time
        
if __name__ == "__main__":
    # Initialize the EasyOCR reader with English language
    #reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a compatible GPU
    latency = 0 # or 200
    input_path = f"../raw_data_parsing/game_frames_{latency}ms/game_frames_{latency}ms"
    output_folder = f"./output_{latency}ms"
    # Define subdirectories
    subdirs = ["ammo", "armour", "health"]

    # Create the full directory paths and ensure they exist
    for subdir in subdirs:
        # Construct the full path (e.g., output/ammo)
        dir_path = os.path.join(output_folder, subdir)
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
    
    OCR_dataprocessor = OCR_DataProcessor(input_path, output_folder)
    OCR_dataprocessor.ocr_label_csv_generation()
    