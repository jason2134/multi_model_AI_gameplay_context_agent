import re
import pandas as pd
from datetime import datetime
import subprocess
import time
import os
import glob
import ast

class OCRDataProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.input_csv = pd.read_csv(input_path)
        self.output_csv = pd.DataFrame(columns=["raw_timestamp","map","game_round","event_type","event_details"])
        
    def transform_df(self):
        df_in = self.input_csv
        df_out = self.output_csv
        df_out['raw_timestamp'] = pd.to_numeric(df_in['file_name'].apply(lambda x: x.split('_')[4].split('.')[0]))
        df_out['map'] = ''
        df_out['game_round'] = ''
        df_out['event_type'] = 'ocr'
        df_out['event_details'] = df_in.apply(lambda row:str({
            'player_ip': '.'.join(row['file_name'].split('_')[0:4]),
            'pred_roi_ammo': row['pred_roi_ammo'],
            'conf_roi_ammo': row['conf_roi_ammo'],
            'pred_roi_health': row['pred_roi_health'],
            'conf_roi_health': row['conf_roi_health'],
            'pred_roi_armour': row['pred_roi_armour'],
            'conf_roi_armour': row['conf_roi_armour'],
            'file_name': row['file_name'],
            'roi_ammo_file_name': row['roi_ammo_file_name'],
            'roi_health_file_name': row['roi_health_file_name'],
            'roi_armour_file_name': row['roi_armour_file_name'],
        }), axis = 1)
        
        df_out.to_csv(self.output_path, index=False)
        
if __name__ ==  "__main__":
    import_dir = './data/raw/ocr_data'
    output_dir = './data/processed/ocr'
    latency = 0
    input_path = f'{import_dir}/ocr_label_{latency}ms.csv'
    output_path = f'{output_dir}/processed_OCR_{latency}ms.csv'
    
    processor = OCRDataProcessor(input_path, output_path)
    processor.transform_df()
        
        