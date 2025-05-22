import os
import cv2
import numpy as np
import pandas as pd
import joblib

class YoloObjectDetection:
    def __init__(self, weights_path = "./yolov4-tiny/yolov4-tiny-custom_final.weights", 
                 config_path = "./yolov4-tiny/yolov4-tiny-custom.cfg", names_path = "./yolov4-tiny/obj.names",
                 ):
        self.weights_path = weights_path
        self.config_path = config_path
        self.names_path = names_path
        
        # Load YOLO
    def load_yolo(self, weights_path, config_path, names_path):
        net = cv2.dnn.readNet(weights_path, config_path)
        with open(names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        return net, classes, output_layers
    
        # Process each frame
    def detect_objects(self, frame, net, output_layers, classes, confidence_threshold=0.5, nms_threshold=0.4):
        height, width = frame.shape[:2]
        
        # Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Initialize lists for detected objects
        class_ids = []
        confidences = []
        boxes = []
        
        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        # Draw bounding boxes
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        detections = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                label = str(classes[class_ids[i]])
                #print(label)
                confidence = confidences[i]
                color = colors[class_ids[i]]
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", 
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'box': [x, y, w, h],
                    'center_x': int(x + w/2),
                    'center_y': int(y + h/2),
                    'x':x,
                    'y':y,
                    'w':w,
                    'h':h,
                })
        
        return frame, detections
    
    def extract_info_from_framesV2(self, folder_path, net, output_layers, classes, chunk_size=50, confidence_threshold=0.5, nms_threshold=0.4):
        frame_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        output_csv = f'yolo_features.csv'  # Added .csv extension for clarity
        if not frame_files:
            print("Error: No frames found in the specified folder.")
            return
        
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(output_csv):
            pd.DataFrame(columns=['player_ip', 'timestamp', 'yolo_obj', 'confidence', 'position_of_object', 
                                'center_x','center_y','x','y','w','h','detected_entities']).to_csv(output_csv, index=False)
        frame_count = 0
        data = []
        for frame_file in frame_files[1:]:
            file_name = os.path.splitext(os.path.basename(frame_file))[0]
            file_name_split = file_name.split('_')
            player_ip = file_name_split[0] + "." + file_name_split[1] + "." + file_name_split[2] + "." + file_name_split[3] # * IP
            timestamp = file_name_split[4] # * timestamp
            frame = cv2.imread(frame_file)
            if frame is None:
                print(f"Warning: Could not read frame {frame_file}")
                continue
            detected_entities = {
                'armour':0,
                'ammo':0,
                'health':0,
                'weapon':0,
                'enemy':0
            }
            embedded_frame, detections = self.detect_objects(frame, net, output_layers, classes)
            for detection in detections:
                detected_entities[detection['label']] += 1
            for detection in detections:
                detection['detected_entities'] = detected_entities
                data.append([player_ip, timestamp, detection['label'], detection['confidence'], detection['box'], 
                             detection['center_x'], detection['center_y'], detection['x'], detection['y'], detection['w'],
                             detection['h'], detection['detected_entities']])
            if len(data) >= chunk_size:
                try:
                    df = pd.DataFrame(data, columns=['player_ip', 'timestamp', 'yolo_obj', 'confidence', 'position_of_object', 
                                                     'center_x','center_y','x','y','w','h','detected_entities'])
                    df.to_csv(output_csv, mode='a', header=False, index=False)  # Append without headers each time
                    data.clear()
                    print(f"Saved chunk of {chunk_size} frames to {output_csv}")
                except Exception as e:
                    print(f"Error saving to CSV: {e}")               
        if data:
            try:
                df = pd.DataFrame(data, columns=['player_ip', 'timestamp', 'yolo_obj', 'confidence', 'position_of_object', 
                                                'center_x','center_y','x','y','w','h','detected_entities'])
                df.to_csv(output_csv, mode='a', header=False, index=False)  # Append without headers each time
                print(f"Saved final {len(data)} frames to {output_csv}")                      
            except Exception as e:
                print(f"Error saving final data to CSV: {e}")
        print(f"Processing complete. Results saved to {output_csv}")
        cv2.destroyAllWindows()                              

if __name__ == "__main__":
    latency = 200
    folder_path = f"../raw_data_parsing/game_frames_{latency}ms/game_frames_{latency}ms"
    yolo_obj_detection = YoloObjectDetection()
    net, classes, output_layers = yolo_obj_detection.load_yolo(yolo_obj_detection.weights_path, yolo_obj_detection.config_path,
                                                               yolo_obj_detection.names_path)
    yolo_obj_detection.extract_info_from_framesV2(folder_path, net, output_layers, classes)


'''
    def extract_info_from_frames(self, folder_path, net, output_layers, classes, chunk_size=50, confidence_threshold=0.5, nms_threshold=0.4):
        frame_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        output_csv = f'yolo_features.csv'  # Added .csv extension for clarity
        if not frame_files:
            print("Error: No frames found in the specified folder.")
            return
        
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(output_csv):
            pd.DataFrame(columns=['player_ip', 'timestamp', 'yolo_obj', 'confidence', 'position_of_object']).to_csv(output_csv, index=False)
        frame_count = 0
        data = []
        
        for frame_file in frame_files[1:]:
            file_name = os.path.splitext(os.path.basename(frame_file))[0]
            file_name_split = file_name.split('_')
            player_ip = file_name_split[0] + "." + file_name_split[1] + "." + file_name_split[2] + "." + file_name_split[3] # * IP
            timestamp = file_name_split[4] # * timestamp
            frame = cv2.imread(frame_file)
            
            if frame is None:
                print(f"Warning: Could not read frame {frame_file}")
                continue
            height, width = frame.shape[:2]
            # Prepare the frame for YOLO
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Process detections
            for out in outs:
                #print(f'Outs: {outs}')
                #print(f'Out: {out}')
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores) 
                    label = classes[class_id] # * object deteced
                    confidence = scores[class_id] # * confidence of object detected
                    obj_box = [0,0,0,0]
                    if confidence > confidence_threshold:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        #x: The x-coordinate of the top-left corner of the rectangle.
                        #y: The y-coordinate of the top-left corner of the rectangle.
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        obj_box = [x, y, w, h] # * area of the box
                        data.append([player_ip, timestamp, label, confidence, obj_box])
                        print(f'Player_IP: {player_ip}, timestamp: {timestamp}, label:{label}, conf:{confidence}, obj_box:{obj_box}')
                    #data.append([player_ip, timestamp, label, confidence, obj_box])
            
            # Write to CSV in chunks
            if len(data) >= chunk_size:
                try:
                    df = pd.DataFrame(data, columns=['player_ip', 'timestamp', 'yolo_obj', 'confidence', 'position_of_object'])
                    df.to_csv(output_csv, mode='a', header=False, index=False)  # Append without headers each time
                    data.clear()
                    print(f"Saved chunk of {chunk_size} frames to {output_csv}")
                except Exception as e:
                    print(f"Error saving to CSV: {e}")
            
            # Save remaining data
        if data:
            try:
                df = pd.DataFrame(data, columns=['player_ip', 'timestamp', 'yolo_obj', 'confidence', 'position_of_object'])
                df.to_csv(output_csv, mode='a', header=False, index=False)  # Append without headers each time
                print(f"Saved final {len(data)} frames to {output_csv}")
            except Exception as e:
                print(f"Error saving final data to CSV: {e}")
        print(f"Processing complete. Results saved to {output_csv}")
        cv2.destroyAllWindows()
'''