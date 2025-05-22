import os
import cv2
import numpy as np
import pandas as pd
import time
#import joblib
from lstm_latency_detection import LatencyDetectionModel  # Explicit import with proper name
from yolo_obj_detection import YoloObjectDetection   # Explicit import with proper name
from ocr_detection import OCR_Detector

class Predictor:
    def __init__(self, video_path, CAPTURE_INTERVAL = 0.016, SLIDING_WINDOW = 2, output_path="./video/output/yolo_ocr_prediction_0ms.avi"):
        self.yolo = YoloObjectDetection()
        self.lstm = LatencyDetectionModel()
        self.ocr = OCR_Detector()
        
        self.video_path = video_path
        self.output_path = output_path
        self.CAPTURE_INTERVAL = CAPTURE_INTERVAL
        self.SLIDING_WINDOW = SLIDING_WINDOW
        

    def process_video(self, video_path,output_path=None):
        # YOLO: Load model
        try:
            print("Loading Yolo...")
            net, classes, output_layers = self.yolo.load_yolo(self.yolo.weights_path, self.yolo.config_path, self.yolo.names_path)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return None

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return None
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_count = 0
        interval = self.CAPTURE_INTERVAL
        start_time = time.time()
        last_capture_time = start_time
        prev_gray = None
        fps = (1 / interval) // 2
        data = []
        # Prepare video writer if output path is specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        while cap.isOpened:
            # Calculate time until next frame should be captured
            current_time = time.time()
            elapsed_since_last = current_time - last_capture_time
            print("Start capturing...")
            
            # Only capture if enough time has elapsed
            if elapsed_since_last >= interval:
                last_capture_time = current_time
                # Capture frame
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"Warning: Could not read frame {frame_count} from video")
                    break
                timestamp = int(current_time * 1000)  # Millisecond precision
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
            
                    diff = cv2.absdiff(prev_gray, gray)
                    _, diff_thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
                    motion_score = np.sum(diff_thresh) / 255
                        
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flow_magnitude = np.linalg.norm(flow, axis=2).mean()
                    data.append({
                        "motion_score": motion_score,
                         "flow_magnitude": flow_magnitude,
                        "time_stamp": timestamp
                    })
                prev_gray = gray
                frame_count += 1
                    
                # Yolo: Perform Predictions
                processed_frame, detections = self.yolo.detect_objects(frame, net, output_layers, classes)
                
                # OCR: HUB Text Extraction
                result_dict = self.ocr.detect_from_frame(frame)
                processed_frame = self.ocr.embed_prediction(processed_frame, result_dict)
                
                
                lstm_predicted_labels = None
                lstm_predictions = None
                
                if len(data) >= self.SLIDING_WINDOW:
                    # Create DataFrame for the current pair
                    df = pd.DataFrame(data)
                    if df.empty:
                        print(f"Skipping window starting at frame {frame_count}: No valid data processed.")
                        return None, None
                    df = self.lstm.compute_differences(df)
                    # Prepare data for model
                    X = self.lstm.prepare_data_for_model(df, ["time_stamp", "motion_score", "motion_score_diff", "flow_magnitude", "flow_magnitude_diff"])
                    print("Making predictions for current window...")
                    lstm_predictions = self.lstm.model.predict(X, verbose=1)
                    lstm_predicted_labels = (lstm_predictions > 0.5).astype(int)
                    del data[0]
                    
                if lstm_predicted_labels is not None and lstm_predictions is not None:
                    final_processed_frame = self.lstm.embed_prediction2frame(processed_frame, lstm_predicted_labels, lstm_predictions)
                else:
                    final_processed_frame = processed_frame
                # Display the frame
                cv2.imshow('YOLO + OCR Detection', final_processed_frame)

                # Write to output video if specified
                if output_path:
                    print("Writing to output video")
                    out.write(final_processed_frame)  # Write frame with both YOLO and LSTM annotations
                # Free memory
                del frame
                del processed_frame
                del final_processed_frame

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")
        return True  # Indicate success

if __name__ == "__main__":
    latency = 0
    video_path = f"./video/Game_record_{latency}ms_V2.mp4"
    output_path = f"./video/output/yolo_ocr_prediction_{latency}ms.mp4"
    model_predict = Predictor(video_path, output_path=output_path)
    model_predict.process_video(video_path, model_predict.output_path)  # Call explicitly