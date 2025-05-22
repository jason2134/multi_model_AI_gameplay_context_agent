import re
import pandas as pd
from datetime import datetime
import subprocess
import time
import os
import glob
import ast
import json

class AggregateDataProcessor:
    def __init__(self, log_input_path, yolo_input_path, ocr_input_path, output_path, context_out_path):
        self.log_input_path = log_input_path
        self.yolo_input_path = yolo_input_path
        self.ocr_input_path = ocr_input_path
        self.output_path = output_path
        self.context_out_path = context_out_path
        
        self.log_csv = pd.read_csv(log_input_path)
        self.yolo_csv = pd.read_csv(yolo_input_path)
        self.ocr_csv = pd.read_csv(ocr_input_path)
        self.ocr_csv['event_details'] = self.ocr_csv['event_details'].apply(self.safe_literal_eval)
        self.yolo_csv['event_details'] = self.yolo_csv['event_details'].apply(self.safe_literal_eval)
        self.log_csv['event_details'] = self.log_csv['event_details'].apply(self.safe_literal_eval)
        self.output_csv = pd.DataFrame(columns=["timestamp","player_ip","map","game_round","event_type","event_details"])
        
        self.item_keys = ['latency', 'item_category', 'item', 'player_id', 'player_ip',
                          'weapon_inventory', 'weapon_num', 'total_armor_picked', 'armor_picked',
                          'total_health_picked', 'health_picked', 'latency_map', 'log_line']
        self.itemLost_keys = ['latency', 'event_type', 'item_category', 'item', 'player_id', 'player_ip', 'weapon_inventory', 'weapon_num',
                              'total_armor_picked', 'armor_picked', 'total_health_picked', 'health_picked', 'latency_map', 'log_line']
        
        self.score_keys = ['latency', 'player_id', 'player_ip', 'points', 'log_line']
        
        self.suicide_keys = ['killer_id', 'victim_id', 'weapon_id', 'killer_ip', 'victim_ip', 'weapon', 'kills', 'deaths', 'kill_dict', 'death_dict',
                             'log_line']
        self.kill_keys = ['latency', 'killer_id', 'victim_id', 'weapon_id', 'killer_ip', 'victim_ip', 'player_id', 'player_ip', 'weapon',
                          'kills', 'deaths', 'kill_dict', 'death_dict', 'log_line']
        self.killed_keys = ['latency', 'killer_id', 'victim_id', 'weapon_id', 'killer_ip', 'victim_ip', 'player_id', 'player_ip', 'weapon', 
                            'kills', 'deaths', 'kill_dict', 'death_dict', 'log_line']
        
        self.yolo_keys = ['player_ip', 'yolo_obj', 'confidence', 'x', 'y', 'width', 'height', 'box_area', 'center_x', 'center_y', 'diff_height', 
                          'position_of_object']
        
        self.ocr_keys = ['player_ip', 'pred_roi_ammo', 'conf_roi_ammo', 'pred_roi_health', 'conf_roi_health', 'pred_roi_armour', 'conf_roi_armour',
                         'file_name', 'roi_ammo_file_name', 'roi_health_file_name', 'roi_armour_file_name']
        
        
    def format_datetime(self, timestamp, mode):
        try:
            ts = float(timestamp)
            if ts < 1e12:  # If it's in seconds, convert to milliseconds
                ts = ts * 1000
            datetime_obj = datetime.fromtimestamp(ts / 1000)  # divide by 1000 back for datetime
            #date_time_str = datetime_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            if mode == 's':
                date_time_str = datetime_obj.strftime('%Y-%m-%d %H:%M:%S')
            elif mode == 'ms':
                date_time_str = datetime_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            else:
                date_time_str = datetime_obj.strftime('%Y-%m-%d %H:%M:%S')
            return date_time_str
        except (ValueError, TypeError, OSError) as e:
            print(f"[Warning] Invalid timestamp '{timestamp}': {e}")
            return f'{None} {None}'
        
    def safe_literal_eval(self,val): # OCR data have issue processing, thats why need to convert first
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return None
        return val  # If it's already a dict or NaN
    
    def get_event_details(self, event, df):
        event_dict = {}
        if event == 'suicide': key_list = self.suicide_keys
        elif event == 'kill':  key_list = self.kill_keys
        elif event == 'score': key_list = self.score_keys
        elif event == 'killed': key_list = self.killed_keys
        elif event == 'item': key_list = self.item_keys
        elif event == 'itemLost': key_list = self.itemLost_keys
        elif event == 'yolo': key_list = self.yolo_keys
        elif event == 'ocr': key_list = self.ocr_keys
        else: 
            print('Invalid Event') 
            return None
        for key in key_list:
            event_dict[key] = df[df['event_type'] == event]['event_details'].apply(lambda d: d.get(key) if isinstance(d, dict) else None)
        return event_dict
    
    def combine_event_details(self,df):
        # per second precision
        # if timestamp is the same --> group all rows with the same timestamp
        #   - for the same event (eg. OCR) --> get only the dominant value count value --> get it into one consistent event comment
        #   - get the different types of events 
        #   - aggregate into one total event
        #   - put true / false statement for any item pick up
        #   - put the obj detections tgh
        parser = ContextParser()
        parsed_df = parser.combine_event_details(df)
        return parsed_df
    
        
    def aggregate(self):
        df = pd.concat([self.log_csv, self.yolo_csv, self.ocr_csv], ignore_index = True) #self.ocr_csv
        df = df.sort_values(by='raw_timestamp') # Sorting by 'Age' in ascending order
        
        df['timestamp_ms'] = df['raw_timestamp'].apply(lambda ts: pd.Series(self.format_datetime(ts, 'ms')))
        df['timestamp_ms'] = pd.to_datetime(df['timestamp_ms'], errors='coerce')
        
        df['timestamp'] = df['raw_timestamp'].apply(lambda ts: pd.Series(self.format_datetime(ts, 's')))
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        df = df.sort_values(by='timestamp_ms')
        
        df['player_ip'] = df['event_details'].apply(lambda d: d.get('player_ip') if isinstance(d, dict) else None)
        df['player_ip'] = df['player_ip'].str.replace('Player_', '', regex=False)
        
        # Validate NewGameRound events
        if df.loc[df['event_type'] == 'NewGameRound', ['map', 'game_round']].isnull().any().any():
            print("Warning: Some NewGameRound events have missing map or game_round")
        # Forward fill
        df['map'] = df['map'].ffill()
        df['game_round'] = df['game_round'].ffill()
        # Handle rows before first NewGameRound (optional)
        df['map'] = df['map'].fillna('unknown')
        df['game_round'] = df['game_round'].fillna(0)
        
        out_df = pd.DataFrame()
        out_df['timestamp'] = df['timestamp']
        out_df['player_ip'] = df['player_ip']
        out_df['map'] = df['map']
        out_df['game_round'] = df['game_round']
        out_df['event_type'] = df['event_type']
        out_df['event_details'] = df['event_details']
        out_df['timestamp_ms'] = df['timestamp_ms']
        out_df.to_csv(self.output_path, index=False)
        
        #----------------- HERE NEEDS CHANGE ----------------------------------------------------------------#
        temp_df = self.combine_event_details(out_df)
        del out_df
        context_out_df = pd.DataFrame()
        context_out_df['timestamp'] = temp_df['timestamp']
        context_out_df['timestamp_ms'] = temp_df['timestamp_ms']
        context_out_df['game_round'] = temp_df['game_round']
        context_out_df['map'] = temp_df['map']
        dict_keys_type = type({}.keys())  # Dynamically get dict_keys type
        context_out_df['events'] = temp_df['events'].apply(lambda d: str(list(d)) if isinstance(d, dict_keys_type) else d)
        
        context_out_df['combinedContext'] = temp_df['combinedContext']
        # Convert details dictionaries to JSON strings to avoid unhashable type error
        context_out_df['combinedContext'] = temp_df['combinedContext'].apply(lambda x: json.dumps(x))
        
        # Drop duplicates
        context_out_df = context_out_df.drop_duplicates(['combinedContext'])

        # Optionally, parse details back to dicts if needed for further processing
        context_out_df['combinedContext'] = context_out_df['combinedContext'].apply(lambda x: json.loads(x))
        
        del temp_df

        context_out_df.to_csv(self.context_out_path, index=False)
        #----------------- HERE NEEDS CHANGE ----------------------------------------------------------------#
        
        
class ContextParser:
    def __init__(self):
        self.map = ''
        self.game_round = 0
        self.latency = 0
        self.suicide_keys = [
            'killer_ip',
            'victim_ip',
            'weapon',
            'log_line'
        ]
        
        self.kill_keys = [
            'killer_ip',
            'victim_ip',
            'player_ip',
            'weapon',
            'kills',
            'deaths',
            'log_line'
        ]
        
        self.score_keys = [
            'player_ip',
            'points',
            'log_line'
        ]
        
        self.killed_keys = [
            'killer_ip',
            'victim_ip',
            'player_ip',
            'weapon',
            'kills',
            'deaths',
            'log_line'
        ]
        
        self.item_keys = [
            'item_category',
            'item',
            'player_ip',
            'weapon_inventory',
            'weapon_num',
            'armor_picked',
            'total_armor_picked',
            'health_picked',
            'total_health_picked',
            'log_line'
        ]
        
        self.itemLost_keys = [
            'item_category',
            'item',
            'player_ip',
            'weapon_inventory',
            'weapon_num',
            'armor_picked',
            'total_armor_picked',
            'health_picked',
            'total_health_picked',
            'log_line'            
        ]
        
        self.yolo_keys = [
            'player_ip',
            'yolo_obj',
            'confidence',
            'x',
            'y',
            'width',
            'height',
            'box_area',
            'center_x',
            'center_y'
        ]
        
        self.ocr_keys = [
            'player_ip',
            'pred_roi_ammo',
            'conf_roi_ammo',
            'pred_roi_health',
            'conf_roi_health',
            'pred_roi_armour',
            'conf_roi_armour'
        ]
        
    def clean_context(self, context):
        # Define all context types and their corresponding key filters
        context_types = {
            'suicide': self.suicide_keys,
            'kill': self.kill_keys,
            'score': self.score_keys,
            'killed': self.killed_keys,
            'item': self.item_keys,
            'itemLost': self.itemLost_keys,
            'yolo': self.yolo_keys,
            'ocr': self.ocr_keys
        }

        for key, key_filter in context_types.items():
            if key in context:
                events = context.get(key, [])
                # Ensure it's a list and filter out None entries
                if isinstance(events, list):
                    context[key] = [
                        {k: v for k, v in event.items() if k in key_filter}
                        for event in events if isinstance(event, dict)
                    ]
                else:
                    context[key] = []  # fallback in case it's not a list

        return context
    
    def combine_event_details(self, df):
        grouped = df.groupby('timestamp_ms')
        for name, group in grouped:
            print(f"Combining context on timestamp_ms {name}")
            
            combined_context = {}

            if 'NewGameRound' in group['event_type'].values:
                new_round_rows = group[group['event_type'] == 'NewGameRound']
                new_map = new_round_rows['map'].unique()[-1]
                new_round = new_round_rows['game_round'].unique()[-1]

                # Extract latency once from the first row with it
                for idx in new_round_rows.index:
                    details = new_round_rows.at[idx, 'event_details']
                    if isinstance(details, dict) and 'latency' in details:
                        self.latency = details['latency']
                        break  # Stop after first found

                combined_context['latency'] = self.latency
                combined_context["map"] = new_map
                combined_context["game_round"] = new_round

                new_round_context = {
                    "NewGameRound": {
                        "game_round": new_round,
                        "map": new_map,
                        "event_type": "NewGameRound",
                        "latency": self.latency
                    }
                }
                df.loc[group.index, 'game_round'] = [new_round]* len(group)
                df.loc[group.index, 'map'] = [new_map] * len(group)
                df.loc[group.index, 'event_type'] = ["NewGameRound"] * len(group)
                df.loc[group.index, 'combinedContext'] = [new_round_context] * len(group)
                del new_round_context
            else:
                for event_type in ["suicide", "kill", "score", "killed", "item", "itemLost", "yolo", "ocr"]:
                    if event_type in group['event_type'].values:
                        rows = group[group['event_type'] == event_type]
                        combined_context[event_type] = []
                        combined_context[event_type].extend(rows['event_details'].tolist())
                        
                df.loc[group.index, 'game_round'] = [new_round] * len(group)
                df.loc[group.index, 'map'] = [new_map] * len(group)
                df.loc[group.index, 'events'] = [combined_context.keys()] * len(group)                        
                df.loc[group.index, 'combinedContext'] = [combined_context] * len(group)
                
            del combined_context
            
        df['combinedContext'] = df['combinedContext'].apply(lambda x: self.clean_context(x) if isinstance(x, dict) else x)
        return df
    
    def simplify_Context_Data(self):
        # Intuition: Simplify the context menu data 
        pass
        

        
        
if __name__ == "__main__":
    import_dir = './data/processed'
    output_dir = './data/processed/final'
    latency = 0
    input_log_path = f'{import_dir}/log/processedLog_{latency}ms.csv'
    input_yolo_path = f'{import_dir}/yolo/processed_Yolo_{latency}ms.csv'
    input_ocr_path = f'{import_dir}/ocr/processed_OCR_{latency}ms.csv'
    
    output_path = f'{output_dir}/aggregated_{latency}ms.csv'
    context_out_path = f'{output_dir}/contextMenu_{latency}ms.csv'
    
    aggregator = AggregateDataProcessor(input_log_path, input_yolo_path, input_ocr_path, output_path, context_out_path)
    aggregator.aggregate()
        