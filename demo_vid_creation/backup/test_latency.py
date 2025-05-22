import cv2
import numpy as np
import time
import gc
from collections import deque
import pandas as pd
from lstm_latency_detection import LatencyDetectionModel

def video_prediction():
    CAPTURE_INTERVAL = 0.032
    latency_detection = LatencyDetectionModel()
    video_path = "./video/Game_record_200ms_V2.mp4"
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return None

    frame_count = 0
    start_time = time.time()
    last_capture_time = start_time
    prev_gray = None
    data = deque(maxlen=2)  # Sliding window of 2
    features = ["time_stamp", "motion_score", "motion_score_diff", "flow_magnitude", "flow_magnitude_diff"]

    while True:
        current_time = time.time()
        elapsed_since_last = current_time - last_capture_time

        if elapsed_since_last >= CAPTURE_INTERVAL:
            last_capture_time = current_time
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Video ended at frame {frame_count}")
                break

            # Downscale frame
            frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Compute motion metrics
                diff = cv2.absdiff(prev_gray, gray)
                _, diff_thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
                motion_score = np.sum(diff_thresh) / 255

                # Always compute optical flow to match preprocess_n_predict_from_frame
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_magnitude = np.linalg.norm(flow, axis=2).mean()

                # Use a relative timestamp to mimic filename-based timing
                timestamp = frame_count  # Or adjust to match training scale (e.g., 1.742e12 + frame_count)
                data.append({
                    "motion_score": motion_score,
                    "flow_magnitude": flow_magnitude,
                    "time_stamp": timestamp
                })

            prev_gray = gray.copy()
            frame_count += 1

            if len(data) == 2:
                # Create DataFrame and compute differences consistently
                df = pd.DataFrame(list(data))
                df = latency_detection.compute_differences(df)
                X = latency_detection.prepare_data_for_model(df, features)

                # Predict
                predictions = latency_detection.model.predict(X, verbose=0)
                predicted_labels = (predictions > 0.5).astype(int)

                # Print results for latest frame
                frame_info = df.iloc[-1]
                pred_label = "0ms" if predicted_labels[-1] == 0 else "200ms"
                confidence = predictions[-1][0]
                print(f"Frame {frame_count}: Motion Score: {frame_info['motion_score']:.2f}, "
                      f"Flow Magnitude: {frame_info['flow_magnitude']:.6f}, "
                      f"Predicted Latency: {pred_label}, Confidence: {confidence:.4f}")

            del frame
            gc.collect()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    video_prediction()