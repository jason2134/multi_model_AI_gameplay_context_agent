import re
import pandas as pd
from datetime import datetime
import subprocess
import time
import os
import glob
import ast

class YoloDataProcessor:
    def __init__(self, input_path,output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.input_csv = pd.read_csv(input_path)
        self.output_csv = pd.DataFrame(columns=["raw_timestamp","map","game_round","event_type","event_details"])
        
    def load_data(self):
        df = self.input_csv
        # Convert types
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['confidence'] = pd.to_numeric(df['confidence'])
        df['position_of_object'] = df['position_of_object'].apply(ast.literal_eval)  # Parse string to list
        # Extract x, y, width, height from position_of_object
        df['x'] = df['position_of_object'].apply(lambda pos: pos[0])
        df['y'] = df['position_of_object'].apply(lambda pos: pos[1])
        df['width'] = df['position_of_object'].apply(lambda pos: pos[2])
        df['height'] = df['position_of_object'].apply(lambda pos: pos[3])
        df['box_area'] = df['width'] * df['height']
        df['center_x'] = df['x'] + df['width'] / 2
        df['center_y'] = df['y'] + df['height'] / 2
        df['diff_height'] = df['height'].diff().fillna(0)
        df['detected_entities'] = df['detected_entities']
        return df
    
    def transformCSV(self,df):
        self.output_csv["raw_timestamp"] = df['timestamp']
        self.output_csv["map"] = ''
        self.output_csv['game_round'] = ''
        self.output_csv['event_type'] = 'yolo'
        self.output_csv['event_details'] = df.apply(lambda row: str({
            'player_ip': row['player_ip'],
            'yolo_obj': row['yolo_obj'],
            'confidence': row['confidence'],
            'x': row['x'],
            'y': row['y'],
            'width': row['width'],
            'height': row['height'],
            'box_area': row['box_area'],
            'center_x': row['center_x'],
            'center_y': row['center_y'],
            'diff_height': row['diff_height'],
            'position_of_object': row['position_of_object'],
            'detected_entities': row['detected_entities']            
        }), axis=1)        
        self.output_csv.to_csv(self.output_path,index=False)
        
if __name__ ==  "__main__":
    import_dir = './data/raw/yolo_data'
    output_dir = './data/processed/yolo'
    latency = 200
    input_path = f'{import_dir}/{latency}ms_yolo_features.csv'
    output_path = f'{output_dir}/processed_Yolo_{latency}ms.csv'
    
    processor = YoloDataProcessor(input_path, output_path)
    csv = processor.load_data()
    processor.transformCSV(csv)