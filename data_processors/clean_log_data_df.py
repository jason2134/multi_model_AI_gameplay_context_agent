import re
import pandas as pd
from datetime import datetime
import subprocess
import time
import os
import glob
import ast
import logging

class rawLog2CSVProcessor:
    def __init__(self, input_path, output_path, latency):
        self.input_path = input_path
        self.output_path = output_path
        self.warm_up_round_complete = False
        self.currentGameRound = 1
        self.current_map = None
        self.current_latency = {}
        self.current_latency_map = {}
        self.data = []
        self.players = {}
        self.kills = {}
        self.deaths = {}
        self.latency = latency
        
    # def format datetime (we will format this later in the aggregating CSV)
    
    '''REPEATED FUNCION'''
    def extract_all_players(self,lines):
        for i, line in enumerate(lines):
            if "Full list of players" in line:
                players = line.split(': ', 2)[2].split(" ") #.split("\n")[0]

                for id, player in enumerate(players):
                    playerIp = player.split('\n')[0]
                    self.players[f'player_{id}'] = f"Player_{playerIp}"

                
    def init_kill_death_score(self):
        for id in self.players:
            self.kills[self.players[id]] = 0
            self.deaths[self.players[id]] = 0
    
    '''REPEATED FUNCION'''
    def checkWarmedUpState(self, lines):
        for i, line in enumerate(lines):
            if "Warmup round complete" in line:
                self.warm_up_round_complete = True
                return i
            
    '''REPEATED FUNCION'''
    def checkCurrentMapCurrentRound(self,line):
            # Extract the map name
            current_map = re.search(r'loaded maps/(.*)\.aas', line).group(1)
            self.currentGameRound += 1  # Increment game round when a new map is loaded
            return current_map, self.currentGameRound
    
    '''REPEATED FUNCION'''
    def parseLatencyMap(self,line):
        current_latency_map = {v: k for k, v in ast.literal_eval(line.split("  ")[1]).items()}
        return current_latency_map
    
    def getKeyfromValue(self, dict, value):
        reversed_dict = {v: k for k, v in dict.items()}
        return reversed_dict.get(value)
    
    def parseKillEvent(self, event, log_line):
        match = re.match(r'Kill: (\d+) (\d+) (\d+): (.+) killed (.+) by (.+)', event)
        if match:
            if "world" in match.group(4): # Removing brackets at <world>
                killed_event, killed_event_info = self.parseVictimKilledEvent(match, log_line)
                killer = "world"
                victim_id = match.group(2)
                if self.getKeyfromValue(self.players, match.group(5)) in self.players.keys():
                    self.deaths[match.group(5)] = self.deaths[match.group(5)] + 1 
                    deaths = self.deaths
                    kills = self.kills
                    event_type = 'suicide'
                    eventInfo = {
                        'killer_id': match.group(1),
                        'victim_id': match.group(2),
                        'weapon_id': match.group(3),
                        'killer_ip': killer,
                        'victim_ip': match.group(5),
                        'weapon': match.group(6),
                        'kills': '',
                        'deaths': '',
                        'kill_dict': kills,
                        'death_dict': deaths,
                        'log_line': log_line
                    }
                    return event_type, eventInfo, killed_event, killed_event_info
            else:
                killer = match.group(4)
                victim = match.group(5)

                if self.getKeyfromValue(self.players, match.group(4)) in self.players.keys():
                #if f"player_{match.group(1)}" in self.players.keys():
                    killer_id = match.group(1)
                    # Update Kills
                    self.kills[killer] = self.kills[killer] + 1
                    kills = self.kills
                    deaths = self.deaths
                    # Append the following params
                    event_type = 'kill'
                    eventInfo = { 
                        'latency': self.latency, # str(self.current_latency[killer.split('_')[1]])
                        'killer_id': match.group(1),
                        'victim_id': match.group(2),
                        'weapon_id': match.group(3),
                        'killer_ip': killer,
                        'victim_ip': victim,
                        'player_id': match.group(1),
                        'player_ip': killer,
                        'weapon': match.group(6),
                        'kills': kills[killer],
                        'deaths': str(deaths[killer]),
                        'kill_dict': str(kills),
                        'death_dict': str(deaths),
                        'log_line': log_line
                    }
                    return event_type, eventInfo, 'None', {}
                
                # Handle if player is victim
                if self.getKeyfromValue(self.players, match.group(5)) in self.players.keys():
                    killed_event, killed_event_info = self.parseVictimKilledEvent(match)
                    victim_id = match.group(2)
                    victim = match.group(5)
                    self.deaths[victim] = self.deaths[victim] + 1
                    deaths = self.deaths
                    event_type = 'killed'
                    eventInfo = {
                        'latency': self.latency,
                        'killer_id': match.group(1),
                        'victim_id': match.group(2),
                        'weapon_id': match.group(3),
                        'killer_ip': match.group(4),
                        'victim_ip': victim,
                        'player_id': match.group(2),
                        'player_ip': victim,
                        'weapon': match.group(6),
                        'kills': kills[victim],
                        'deaths': str(deaths[victim]),
                        'kill_dict': str(kills),
                        'death_dict': str(deaths),
                        'log_line': log_line            
                    }
                    return event_type, eventInfo, killed_event, killed_event_info
                    
                else: # Bot kills
                    return 'None', {}, 'None', {}
        return 'None', {}, 'None', {}
    
    def parseVictimKilledEvent(self, event_match, log_line):
        match = event_match
        victimId = match.group(2)
        victimIP = match.group(5).split(" ")[0]
        player_id_key = f'player_{victimId}'
        if self.getKeyfromValue(self.players, victimIP) in self.players.keys():
        #if player_id_key in self.players.keys():
            if self.players[player_id_key] == victimIP:
                self.inventory.inventory_player_dead_event(victimId)
                weapon_inventory = self.inventory.weapon_inventory[player_id_key]
                
                total_armor_picked = 0
                armor_picked = self.inventory.armor_inventory[player_id_key]
                
                total_health_picked = 0
                health_picked = self.inventory.health_inventory[player_id_key]
                
                eventType = 'itemLost'
                eventInfo = {
                    'latency': self.latency,
                    'event_type': "itemLost",
                    'item_category': "",
                    'item': "",
                    'player_id': player_id_key,
                    'player_ip': self.players[player_id_key],
                    'weapon_inventory': weapon_inventory,
                    'weapon_num': len(weapon_inventory),
                    'total_armor_picked':total_armor_picked,
                    'armor_picked':armor_picked,
                    'total_health_picked':total_health_picked,
                    'health_picked':health_picked,
                    'latency_map': str(self.current_latency_map),
                    'log_line': log_line
                }
                return eventType, eventInfo
        return 'None', {}
        
    
    def parse_playerscore(self,event, log_line):
        match = re.match(r'PlayerScore: (\d+) (\d+): (.+) now has (\d+) points', event)
        
        if match:
            event_type = 'score'
            if self.getKeyfromValue(self.players, match.group(3)) in self.players.keys():
                return event_type, {
                    'latency': self.latency,
                    'player_id': match.group(1),
                    #'score': match.group(2),
                    'player_ip': match.group(3),
                    'points': match.group(4),
                    'log_line': log_line
                }
        return "None", {}
    
    def parseItemEvent(self, event, log_line):
        match = re.match(r'Item: (\d+) (.+)', event)
        #print(f'match: {match}')
        if match:
            match_dict =  {
                "playerId": match.group(1),
                "item": match.group(2)
            }
            player_id_key = f"player_{match_dict['playerId']}"
            #print(player_id_key)
            if player_id_key in self.players.keys():
                if 'weapon' in match_dict["item"]:
                    current_inventory = dict(self.inventory.weapon_inventory[player_id_key])
                    current_inventory = self.inventory.update_inventory(player_id_key, current_inventory, match_dict["item"])
                    weapon_inventory = current_inventory
                                
                    total_armor_picked = self.inventory.total_armor[player_id_key]
                    armor_picked = self.inventory.armor_inventory[player_id_key]
                                
                    total_health_picked = self.inventory.total_health[player_id_key]
                    health_picked = self.inventory.health_inventory[player_id_key]
                    
                elif 'armor' in match_dict["item"]:
                    current_inventory = self.inventory.get_inventory(self.inventory.armor_inventory[player_id_key])
                    current_inventory = self.inventory.update_inventory(player_id_key, current_inventory, match_dict["item"])
                    weapon_inventory = self.inventory.weapon_inventory[player_id_key]
                                
                    self.inventory.total_armor[player_id_key] = self.inventory.calculate_total_health_or_armor_picked(current_inventory,'armor')
                    total_armor_picked = self.inventory.total_armor[player_id_key]
                    armor_picked = current_inventory
                                
                    total_health_picked = self.inventory.total_health[player_id_key]
                    health_picked = self.inventory.health_inventory[player_id_key]
                    
                elif 'health' in match_dict["item"]:
                    current_inventory = self.inventory.get_inventory(self.inventory.health_inventory[player_id_key])
                    current_inventory = self.inventory.update_inventory(player_id_key, current_inventory, match_dict["item"])
                    weapon_inventory = self.inventory.weapon_inventory[player_id_key]
                                
                    total_armor_picked = self.inventory.total_armor[player_id_key]
                    armor_picked = self.inventory.armor_inventory[player_id_key]
                                
                    self.inventory.total_health[player_id_key] = self.inventory.calculate_total_health_or_armor_picked(current_inventory,'health')
                    total_health_picked = self.inventory.total_health[player_id_key]
                    health_picked = self.inventory.health_inventory[player_id_key]
                    
                else:
                    current_inventory = self.inventory.get_inventory(self.inventory.health_inventory[player_id_key])
                    current_inventory = self.inventory.update_inventory(player_id_key, current_inventory, match_dict["item"])
                    weapon_inventory = self.inventory.weapon_inventory[player_id_key]
                                
                    total_armor_picked = self.inventory.total_armor[player_id_key]
                    armor_picked = self.inventory.armor_inventory[player_id_key]
                                
                    total_health_picked = self.inventory.total_health[player_id_key]
                    health_picked = self.inventory.health_inventory[player_id_key]
                event_type = 'item'
                item_event = {
                    'latency':self.latency,
                    'item_category': self.create_item_category(match_dict["item"]),
                    'item': match_dict["item"],
                    'player_id': player_id_key,
                    'player_ip': self.players[player_id_key],
                    'weapon_inventory': weapon_inventory,
                    'weapon_num': len(weapon_inventory),
                    'total_armor_picked': total_armor_picked,
                    'armor_picked': armor_picked,
                    'total_health_picked': total_health_picked,
                    'health_picked': health_picked,
                    'latency_map': str(self.current_latency_map),
                    'log_line': log_line
                }
                return event_type, item_event
        return 'None', {}
                                      
    def create_item_category(self, item):
        if "health" in item:
            return "Health"
        elif "ammo" in item:
            return "Ammo"
        elif "weapon" in item:
            return "Weapon"
        elif "armor" in item:
            return "Armor"
        else:
            return "Misc"
    
    def processRawLog(self):
        last_loaded_map = None
        with open(self.input_path, 'r') as file:
            lines = file.readlines()
        
        self.extract_all_players(lines)
        self.inventory = InventoryManager(self.players)
        self.init_kill_death_score()
        #start_index = self.checkWarmedUpState(lines)
        
        for line in lines:
            if 'loaded maps/' in line:
                last_loaded_map = line.split('loaded maps/')[1].split('.aas')[0]
        self.current_map = last_loaded_map
        
        with open(self.output_path, 'w') as output_file:
            for line in lines:
                try:
                    timestamp, events = line.split(': ', 1)
                except ValueError:
                    logging.warning(f"Skipping malformed line: {line.strip()}")
                    continue
                
                if 'loaded maps/' in events:
                    self.current_map, self.currentGameRound = self.checkCurrentMapCurrentRound(line)
                    self.init_kill_death_score()
                    
                    merged_event = {
                        'raw_timestamp': timestamp,
                        'map': self.current_map,
                        'game_round': self.currentGameRound,
                        'event_type': 'NewGameRound',
                        'event_details': {'latency': self.latency},                        
                    }
                    self.data.append(merged_event)

                elif 'Latency map' in events:
                    try:
                        self.current_latency = ast.literal_eval(line.split(":  ")[1])
                        self.current_latency_map = self.parseLatencyMap(line)

                    except (SyntaxError, ValueError):
                       logging.warning(f"Failed to parse latency map: {line.strip()}")
                else:
                    merged_event = {
                        'raw_timestamp': timestamp,
                        'map': self.current_map,
                        'game_round': self.currentGameRound,
                        'event_type': '',
                        'event_details': '',
                        #'log_line': line.strip()
                    }
                    log_line = line.strip()
                    if 'Kill' in events:
                        event_type, parsed_event, victimEvent, parsedVictimEvent = self.parseKillEvent(events, log_line)

                        if victimEvent != 'None':
                            victim_merged_event = merged_event.copy()
                            victim_merged_event['event_type'] = victimEvent
                            victim_merged_event['event_details'] = parsedVictimEvent
                            self.data.append(victim_merged_event)

                    elif 'PlayerScore:' in events:
                        event_type, parsed_event = self.parse_playerscore(events, log_line)

                    elif 'Item' in events:
                        event_type, parsed_event = self.parseItemEvent(events, log_line)

                    else:
                        #logging.debug(f"Unrecognized event: {events.strip()}")
                        continue
                    
                    if event_type != 'None':
                        merged_event['event_type'] = event_type
                        merged_event['event_details'] = parsed_event
                        self.data.append(merged_event)

        
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path, index=False)
        logging.info(f"Saved {len(self.data)} events to {self.output_path}")     
                    
                    
class InventoryManager:
    def __init__(self, playerMapping):
        self.weapon_inventory = {}
        self.armor_inventory = {}
        self.health_inventory = {}
        self.total_health = {}
        self.total_armor = {}
        self.playerMapping = playerMapping
        self.init_inventory()
        
    def init_inventory(self):
        # Initialize each player's inventory with just a machinegun
        for playerId in self.playerMapping:
            self.weapon_inventory[playerId] = {"weapon_machinegun": 1}
            self.armor_inventory[playerId] = {}
            self.health_inventory[playerId] = {}
            self.total_health[playerId] = 0
            self.total_armor[playerId] = 0
            
    def get_inventory(self,inventory):
        if inventory== 1:
            return {}
        else:
            return dict(inventory)
        
    def inventory_player_dead_event(self, victimId):
        # Reset the dead player's inventory to just a machinegun
        player_id_key = f'player_{victimId}'
        self.weapon_inventory[player_id_key] = {"weapon_machinegun": 1}
        self.armor_inventory[player_id_key] = {}
        self.health_inventory[player_id_key] = {}
        self.total_health[player_id_key] = 0
        self.total_armor[player_id_key] = 0
        
    def update_inventory(self, player_id_key, current_inventory,item):
        if item in current_inventory:
            current_inventory[item] += 1
            #current_inventory[weapon] += 1
        else:
            current_inventory[item] = 1
            
        if 'weapon' in item:
            self.weapon_inventory[player_id_key] = current_inventory
        elif 'armor' in item:
            self.armor_inventory[player_id_key] = current_inventory
        elif 'health' in item:
            self.health_inventory[player_id_key] = current_inventory
        
        return dict(current_inventory)  # Return a copy of the updated inventory
    
    def calculate_total_health_or_armor_picked(self,inventory,item_type: str):
        if len(inventory) == 0:
            return 0
        if item_type == 'health':
            small = 0
            medium = 0
            large = 0
            mega = 0
            for key in inventory.keys():
                if 'small' in key:
                    small = inventory[key] * 25
                elif 'item_health' == key:
                    medium = inventory[key] * 50
                elif 'large' in key:
                    large = inventory[key] * 100
                elif 'mega' in key:
                    mega = inventory[key] * 200
                return small + medium + large + mega
        elif item_type == 'armor':
            shard = 0
            body = 0
            combat = 0
            heavy = 0
            for key in inventory.keys():
                if 'shard' in key:
                    shard = inventory[key] * 5
                elif 'body' in key:
                    body = inventory[key] * 25
                elif 'combat' in key:
                    combat = inventory[key] * 50
                elif 'heavy' in key:
                    heavy = inventory[key] * 100
                return shard + body + combat + heavy
            
        
if __name__ == "__main__":
    import_dir = './data/raw/log_data'
    output_dir = './data/processed/log'
    '''
    log_files = glob.glob(os.path.join(import_dir, '*.log'))

    if not log_files:
        raise FileNotFoundError("No .log file found in the import directory.")

    if len(log_files) > 1:
        print("Warning: Multiple .log files found. Using the first one.")    
    '''
    latency = 200
    input_path = f'{import_dir}/game_log_{latency}ms.log'
    output_path = f'{output_dir}/processedLog_{latency}ms.csv'
    
    processor = rawLog2CSVProcessor(input_path, output_path, latency)
    processor.processRawLog()
    