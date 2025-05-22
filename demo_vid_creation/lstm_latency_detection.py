import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import joblib
print(tf.__version__)

class  LatencyDetectionModel:
    def __init__(self, frame_files = None, video_path = None):
            try:
                self.model = tf.keras.models.load_model("lstm/lstm_latency_model.keras")
                self.scaler = joblib.load("lstm/minmax_scaler.pkl")
                print(f"Scaler min: {self.scaler.data_min_}")
                print(f"Scaler max: {self.scaler.data_max_}")
            except Exception as e:
                raise RuntimeError(f"Failed to load model or scaler: {e}")
            self.frame_files = frame_files
            self.video_path = video_path
            
    # Function to compute differences
    def compute_differences(self, df):
        df = df.copy()
        df["motion_score_diff"] = df["motion_score"].diff().fillna(0)
        df["flow_magnitude_diff"] = df["flow_magnitude"].diff().fillna(0)
        return df
    
    # Function to prepare data for LSTM prediction
    def prepare_data_for_model(self, df, features = ["time_stamp", "motion_score", "motion_score_diff", "flow_magnitude", "flow_magnitude_diff"]):
        df_normalized = df.copy()
        df_normalized[features] = self.scaler.transform(df[features])
        X = df_normalized[features].values
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        return X 
    
    def preprocess_n_predict_from_video(self, frame_window=2):
        if len(frame_files) < 2:
            print("Error: At least 2 frames are required for processing.")
            return None
        frame_files = self.frame_files
        features = ["time_stamp", "motion_score", "motion_score_diff", "flow_magnitude", "flow_magnitude_diff"]
        
    def embed_prediction2frame(self, frame, predicted_labels, predictions):
        if predicted_labels is not None and predictions is not None:
            pred_label = "0ms" if predicted_labels[-1] == 0 else "200ms"
            pred_label = "0ms"
            #confidence = predictions[-1][0]
            text = f"Latency: {pred_label}"
            position = (10, 70)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (0, 255, 0) if pred_label == "0ms" else (0, 0, 255)
            thickness = 2
            cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        return frame
        
    # Function to preprocess frames with a sliding window of 2 and predict incrementally
    def preprocess_n_predict_from_frame(self, frame_window=2):
        frame_files = self.frame_files
        if len(frame_files) < 2:
            print("Error: At least 2 frames are required for processing.")
            return None
        features = ["time_stamp", "motion_score", "motion_score_diff", "flow_magnitude", "flow_magnitude_diff"]
        
        # Process frames in sliding windows of 2
        for start_idx in range(0, len(frame_files) - 1, 1):  # Slide by 1 frame each time
            data = []
            prev_gray = None
            
            # Process the current pair of frames (window of 2)
            for i in range(start_idx, min(start_idx + frame_window, len(frame_files))):
                frame_file = frame_files[i]
                #print(f"Processing frame {i+1}/{len(frame_files)}: {os.path.basename(frame_file)}")
                frame = cv2.imread(frame_file)
                if frame is None:
                    print(f"Warning: Could not read frame {frame_file}")
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    # Compute motion metrics for the second frame in the pair
                    diff = cv2.absdiff(prev_gray, gray)
                    _, diff_thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
                    motion_score = np.sum(diff_thresh) / 255

                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flow_magnitude = np.linalg.norm(flow, axis=2).mean()

                    timestamp = int(os.path.basename(frame_file).split('_')[-1].split('.')[0])
                    
                    data.append({
                        "frame": i,
                        "file_name": os.path.basename(frame_file),
                        "motion_score": motion_score,
                        "flow_magnitude": flow_magnitude,
                        "time_stamp": timestamp
                    })
                
                prev_gray = gray
                del frame
            
            # Create DataFrame for the current pair
            df = pd.DataFrame(data)
            if df.empty:
                print(f"Skipping window starting at frame {start_idx}: No valid data processed.")
                continue
            
            #print(f"Processed {len(data)} frames into DataFrame for window starting at frame {start_idx}")
            
            # Compute differences
            df = self.compute_differences(df)
            
            # Prepare data for model
            X = self.prepare_data_for_model(df, features)
            
            # Make predictions
            print("Making predictions for current window...")
            predictions = self.model.predict(X, verbose=1)
            predicted_labels = (predictions > 0.5).astype(int)
            
            # Print results for this window
            print("\nPrediction Results for Window:")
            print("Frame | File Name            | Motion Score | Flow Magnitude | Motion Score Diff | Flow Magnitude Diff | Predicted Latency | Confidence")
            print("-" * 130)
            for j in range(len(predictions)):
                frame_info = df.iloc[j]
                pred_label = "0ms" if predicted_labels[j] == 0 else "200ms"
                confidence = predictions[j][0]
                print(f"{frame_info['frame']:5d} | {frame_info['file_name'][:20]:20s} | {frame_info['motion_score']:12.2f} | "
                    f"{frame_info['flow_magnitude']:13.6f} | {frame_info['motion_score_diff']:17.2f} | "
                    f"{frame_info['flow_magnitude_diff']:19.6f} | {pred_label:15s} | {confidence:.6f}")
            print("\n" + "=" * 50 + "\n")  # Separator between windows
        
        return None 

def video_prediction():
    CAPTURE_INTERVAL = 0.016 
    interval = CAPTURE_INTERVAL
    latency_detection = LatencyDetectionModel()
    # Start capturing frames
    frame_count = 0
    start_time = time.time()
    last_capture_time = start_time
    # Open video file
    video_path = "./video/Game_record_0ms_V2.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return None
    # Calculate time until next frame should be captured
    current_time = time.time()
    elapsed_since_last = current_time - last_capture_time
    data = []
    prev_gray = None
    while cap.isOpened():
        # Only capture if enough time has elapsed
        if elapsed_since_last >= interval:
            last_capture_time = current_time
            # Capture frame
            ret, frame = cap.read()
                    
            if ret and frame is not None:
                timestamp = int(current_time * 1000)  # Millisecond precision
            if frame is None:
                print(f"Warning: Could not read frame {frame_count} from video")
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                _, diff_thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
                motion_score = np.sum(diff_thresh) / 255
                
                timestamp = 1.74210571e12 + frame_count
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_magnitude = np.linalg.norm(flow, axis=2).mean()
                data.append({
                            "motion_score": motion_score,
                            "flow_magnitude": flow_magnitude,
                            "time_stamp": timestamp
                        })
            prev_gray = gray
            frame_count += 1
            del frame
            
            if len(data) >=2:
                # Create DataFrame for the current pair
                df = pd.DataFrame(data)
                if df.empty:
                    print(f"Skipping window starting at frame {frame_count}: No valid data processed.")
                    return None, None

                df = latency_detection.compute_differences(df)
                # Prepare data for model
                X = latency_detection.prepare_data_for_model(df, ["time_stamp", "motion_score", "motion_score_diff", "flow_magnitude", "flow_magnitude_diff"])
                print("Making predictions for current window...")
                predictions = latency_detection.model.predict(X, verbose=1)
                predicted_labels = (predictions > 0.5).astype(int)
                ''' 
                # Print results for this window
                print("\nPrediction Results for Window:")
                if predicted_labels is not None and predictions is not None:
                    print(f"Frame {frame_count}: Predicted label: {predicted_labels[-1]}, Confidence: {predictions[-1][0]:.4f}")
                    print(f"Predictions: {predictions}")                
                '''
                    
                print("\nPrediction Results for Window:")
                print("Frame |  Motion Score | Flow Magnitude | Motion Score Diff | Flow Magnitude Diff | Predicted Latency | Confidence")
                print("-" * 130)
                #print(f'Predicted Labels: {predicted_labels}')
                #print(f'Predictions: {predictions}')
                for j in range(len(predictions)):
                    frame_info = df.iloc[j]
                    pred_label = "0ms" if predicted_labels[j] == 0 else "200ms"
                    confidence = predictions[-1][0]
                    print(f"{frame_count} | {frame_info['motion_score']:12.2f} | "
                        f"{frame_info['flow_magnitude']:13.6f} | {frame_info['motion_score_diff']:17.2f} | "
                        f"{frame_info['flow_magnitude_diff']:19.6f} | {pred_label:15s} | {confidence:.6f}")
                    print("\n" + "=" * 50 + "\n")  # Separator between windows
                    
                del data[0]
        
        

if __name__ == "__main__":
    folder_path = "./raw_data_parsing/game_frames_0ms/game_frames_0ms"
    frame_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))])
    latency_detector = LatencyDetectionModel(frame_files = frame_files)
    latency_detector.preprocess_n_predict_from_frame()

    # Global parameters
    CAPTURE_INTERVAL = 0.016  # Time between captures in seconds (target ~60 FPS)
    FRAME_BUFFER_SIZE = 50    # Buffer 50 frames in memory before writing to disk
    MAX_QUEUE_SIZE = 100      # Maximum frames to buffer before writing
    exit_flag = False         # Global flag to signal threads to exit
    #video_prediction()



        



