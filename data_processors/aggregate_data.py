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
                          'position_of_object', 'detected_entities']
        
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
    
    def simplify_Context_df(self,df):
        parser = ContextParser()
        parsed_df = parser.simplify_Context_df(df)
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
        
        simp_context_df = self.simplify_Context_df(df)
        simp_context_df.to_csv(self.context_out_path, index=False)
        
class ContextParser:
    def __init__(self, latency = 0):
        self.map = ''
        self.game_round = 0
        self.latency = latency
        self.player_inventory = {
            "weapon": {"weapon_machinegun": 1},
            "armor": 0,
            "health": 0
        }
        
    def simplify_Context_df(self, df):
        grouped = df.groupby('timestamp')
        data = []
        for timestamp, group in grouped:
            combined_context = {
                "timestamp": str(timestamp),
                "player_ammo": 0,
                "player_health": 0,
                "player_armor": 0,
                "game_round": self.game_round,
                "map": self.map,
                "player_sight_event":"",
                "detected_entities": (),
                "suicide_event": "",
                "kill_event": "",
                "score_event": "",
                "killed_event": "",
                "itemPicked_event": [],
                "itemLost_event": "",
                "newGameRound_event": "",
                "player_inventory": self.player_inventory,
                "log_line": ""
            }
            
            for event_type, temp in group.groupby('event_type'):
                flattened_temp = temp['event_details'].apply(pd.Series)
                
                if event_type == 'ocr':
                    if flattened_temp.isnull().all().all():
                        continue  # OCR totally failed
                    for key, conf_key, pred_key in [
                        ("player_ammo", "conf_roi_ammo", "pred_roi_ammo"),
                        ("player_health", "conf_roi_health", "pred_roi_health"),
                        ("player_armor", "conf_roi_armour", "pred_roi_armour"),
                    ]:
                        series = flattened_temp[flattened_temp[conf_key] > 0.6][pred_key]
                        combined_context[key] = series.value_counts().idxmax() if not series.empty else 0
                
                elif event_type == 'yolo':
                    # Filter YOLO events with confidence > 0.5
                    flattened_temp_yolo = flattened_temp[flattened_temp['confidence'] > 0.6]

                    if not flattened_temp_yolo.empty:
                        # Ensure group contains only YOLO events for timestamp extraction
                        yolo_group = group[group['event_type'] == 'yolo']
                        if not yolo_group.empty:
                            # Get first and last timestamps
                            first_ms = yolo_group['timestamp_ms'].iloc[0].round('ms')
                            last_ms = yolo_group['timestamp_ms'].iloc[-1].round('ms')
                            print(f'first_ms: {first_ms}, last_ms: {last_ms}')

                            # Reset indices to avoid misalignment
                            flattened_temp_yolo = flattened_temp_yolo.reset_index(drop=True)
                            ms_values = yolo_group[['timestamp_ms']].reset_index(drop=True)

                            # Ensure lengths match before concatenation
                            min_len = min(len(flattened_temp_yolo), len(ms_values))
                            flattened_temp_yolo = flattened_temp_yolo.iloc[:min_len]
                            ms_values = ms_values.iloc[:min_len]

                            # Add timestamp_ms to filtered YOLO dataframe
                            flattened_temp_yolo = pd.concat([flattened_temp_yolo, ms_values], axis=1)
                            flattened_temp_yolo['timestamp_ms'] = flattened_temp_yolo['timestamp_ms'].dt.round('ms')

                            # Drop rows with NaT in timestamp_ms
                            flattened_temp_yolo = flattened_temp_yolo.dropna(subset=['timestamp_ms'])

                            # Filter rows for first_ms and last_ms
                            filtered_rows = flattened_temp_yolo[
                                (flattened_temp_yolo['timestamp_ms'] == first_ms) |
                                (flattened_temp_yolo['timestamp_ms'] == last_ms)
                            ]

                            grouped_by_ms = filtered_rows.groupby('timestamp_ms')
                            sight_event_array = []

                            for ms, rows_at_ms in grouped_by_ms:
                                detections_at_ms = []
                                for _, row in rows_at_ms.iterrows():
                                    # Skip rows with missing yolo_obj
                                    if pd.isna(row['yolo_obj']):
                                        continue
                                    print(f'OBJect: {row["yolo_obj"]}, MS: {row["timestamp_ms"]}')
                                    ''' 
                                    detected_entity = {
                                        'obj': row['yolo_obj'],
                                        'conf': row['confidence'],
                                        'center_x': row['center_x'],
                                        'center_y': row['center_y'],
                                    }
                                    combined_context['player_sight_details'].append(detected_entity)
                                    '''
                                    detections_at_ms.append(f"{row['yolo_obj']} at ({row['center_x']}, {row['center_y']})")

                                if detections_at_ms:  # Only append if there are valid detections
                                    group_str = f"at {ms}ms: " + ", ".join(detections_at_ms)
                                    sight_event_array.append(group_str)

                            # Store detected entities for first and last timestamps
                            if not filtered_rows.empty:
                                combined_context['detected_entities'] = (
                                    filtered_rows['detected_entities'].iloc[0],
                                    filtered_rows['detected_entities'].iloc[-1]
                                )
                                combined_context['player_sight_event'] = f"Player saw {'; '.join(sight_event_array)}"
                            else:
                                print("No valid YOLO detections after filtering")
                        else:
                            print("No YOLO events in the group")
                    else:
                        print("No YOLO detections with confidence > 0.6")

                    
                elif event_type == 'suicide':
                    if not flattened_temp.empty:
                        combined_context["suicide_event"] = (
                            f'{flattened_temp.iloc[0]["victim_ip"]} died by fall off from game map'
                        )
                        combined_context["log_line"] = flattened_temp.iloc[0]['log_line']
                
                elif event_type == 'kill':
                    for _, row in flattened_temp.iterrows():
                         combined_context["kill_event"] = [
                            f'{row["killer_ip"]} killed enemy {row["victim_ip"]} by {row["weapon"]}, player currently has {row["kills"]} kills and {row["deaths"]} deaths'
                         ]
                         combined_context['log_line'] = [row['log_line']]
                    
                    if len(combined_context["kill_event"]) == 1:
                        combined_context["kill_event"] = combined_context["kill_event"][0]
                        combined_context['log_line'] = combined_context["log_line"][0]
                        
                elif event_type == 'score':
                    for _, row in flattened_temp.iterrows():
                        combined_context["score_event"] = [
                        f'{row["player_ip"]} score updated to {row["points"]}'
                        ]
                        combined_context['log_line'] = [row['log_line']]
                        
                    if len(combined_context["score_event"]) == 1:
                        combined_context["score_event"] = combined_context["score_event"][0]
                        combined_context['log_line'] = combined_context["log_line"][0]
                        
                elif event_type == 'killed':
                    if not flattened_temp.empty:
                        combined_context["killed_event"] = (
                            f'{flattened_temp.iloc[0]["victim_ip"]} killed by enemy {flattened_temp.iloc[0]["killer_ip"]} by {flattened_temp.iloc[0]["weapon"]}, '
                            f'player currently has {flattened_temp.iloc[0]["kills"]} kills and {flattened_temp.iloc[0]["deaths"]} deaths'
                        )
                        combined_context['log_line'] = flattened_temp.iloc[0]['log_line']
                        
                elif event_type == 'item':
                    for _, row in flattened_temp.iterrows():
                        combined_context["itemPicked_event"] = [f'{row["player_ip"]} picked up {row["item_category"]} type: {row["item"]}']
                        combined_context['log_line'] = [row['log_line']]
                        self.player_inventory = {
                            "weapon": row["weapon_inventory"],
                            "armor": row["armor_picked"],
                            "health": row["health_picked"]
                        }
                    combined_context["player_inventory"] = self.player_inventory
                    
                    if len(combined_context["itemPicked_event"]) == 1:
                        combined_context["itemPicked_event"] = combined_context["itemPicked_event"][0]
                        combined_context['log_line'] = combined_context["log_line"][0]
                
                elif event_type == 'itemLost':
                    if not flattened_temp.empty:
                        self.player_inventory = {
                            "weapon": {"weapon_machinegun": 1},
                            "armor": 0,
                            "health": 0
                        }
                        combined_context["player_inventory"] = self.player_inventory
                        combined_context["itemLost_event"] = (
                            f'{flattened_temp.iloc[0]["player_ip"]} lost all items due to killed or suicide event.'
                        )
                        combined_context['log_line'] = flattened_temp.iloc[0]['log_line']
                        
                elif event_type == 'NewGameRound':
                    if not temp.empty:
                        new_round_info = temp.iloc[0]
                        self.game_round = new_round_info["game_round"]
                        self.map = new_round_info["map"]
                        self.player_inventory = {
                            "weapon": {"weapon_machinegun": 1},
                            "armor": 0,
                            "health": 0
                        }
                        
                        combined_context.update({
                            "player_inventory": self.player_inventory,
                            "map": self.map,
                            "game_round": self.game_round,
                            "newGameRound_event": (
                                f'New game round started. Changed game round to {self.game_round} '
                                f'and map to {self.map}. Player inventory, kills, and deaths are reset.'
                            )
                        })
            data.append(combined_context)
            
        return pd.DataFrame(data)
    
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

                        
                    