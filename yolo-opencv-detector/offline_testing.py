import cv2
import numpy as np

# Load YOLO
def load_yolo(weights_path, config_path, names_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes, output_layers

# Process each frame
def detect_objects(frame, net, output_layers, classes, confidence_threshold=0.5, nms_threshold=0.4):
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
            confidence = confidences[i]
            color = colors[class_ids[i]]
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            detections.append({
                'label': label,
                'confidence': confidence,
                'box': [x, y, w, h]
            })
    
    return frame, detections

def process_video(video_path, weights_path, config_path, names_path, output_path=None):
    # Load the YOLO model
    net, classes, output_layers = load_yolo(weights_path, config_path, names_path)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Prepare video writer if output path is specified
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects in current frame
        processed_frame, detections = detect_objects(frame, net, output_layers, classes)
        
        # Display the frame
        cv2.imshow('YOLO Object Detection', processed_frame)
        
        # Write to output video if specified
        if output_path:
            out.write(processed_frame)
        
        frame_count += 1
        print(f"Processed frame {frame_count}")
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")

# Example usage
if __name__ == "__main__":
    # Specify your file paths
    video_path = "test_obj_video_0ms.mp4"
    weights_path = "./yolov4-tiny-custom_final.weights"
    config_path = "./yolov4-tiny/yolov4-tiny-custom.cfg"
    names_path = "./yolov4-tiny/obj.names"
    output_path = "test_video.avi"  # Optional
    
    process_video(video_path, weights_path, config_path, names_path, output_path)