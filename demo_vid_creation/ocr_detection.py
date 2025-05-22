import easyocr
import cv2
import numpy as np
import os
import csv
import glob
import random
import pandas as pd

class OCR_Detector():
    def __init__(self):
        self.roi = {
            'ammo': (0.184150,0.945098, 0.122549, 0.109804),
            'health': (0.398366,0.948366,0.121569,0.103268),
            'armour': (0.616993, 0.945098, 0.117647, 0.109804)
        }
        self.reader = easyocr.Reader(['en'], gpu=False) 
        
    def crop_image(self, frame, param):
        if frame is None:
            print(f"Error: Could not load frame from CV")
            return None
        height, width = frame.shape[:2]
        print(f"\nProcessing frame - Dimensions: {width}x{height}")
        x_center, y_center, roi_width, roi_height = self.roi[param]
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
        cropped_image = frame[y1:y2, x1:x2]
        
        return cropped_image
    
    def detect_from_frame(self, frame):
        result_dict = {}
        for param in self.roi.keys():
            cropped_image = self.crop_image(frame, param)
            cropped_image = cv2.convertScaleAbs(cropped_image, alpha=2.0, beta=0) # beta = 10
            
            results = self.reader.readtext(cropped_image, detail=1, paragraph=False,  allowlist='0123456789')
            if not results:
                cropped_image = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                results = self.reader.readtext(cropped_image, detail=1, paragraph=False, allowlist='0123456789')
                if not results:
                    cropped_image = cv2.resize(cropped_image, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
                    results = self.reader.readtext(cropped_image, detail=1, paragraph=False, allowlist='0123456789')
                    
            if results:
                for (bbox, text, prob) in results:
                    result_dict[f'pred_roi_{param}'] = text
                    result_dict[f'conf_roi_{param}']  = prob
            else:
                #data.append([cropped_image_name, 'No text detected', 0.0])
                result_dict[f'pred_roi_{param}'] = 'No text detected'
                result_dict[f'conf_roi_{param}']  = 0.0    

        return result_dict
                    
    def embed_prediction(self, frame, result_dict):
        if result_dict is not None:
            text = f"Ammo: {result_dict['pred_roi_ammo']} | HP: {result_dict['pred_roi_health']} | Armour: {result_dict['pred_roi_armour']}"
            position = (10, 30)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (255, 255, 255)
            thickness = 2
            cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        return frame
    

                
        
        