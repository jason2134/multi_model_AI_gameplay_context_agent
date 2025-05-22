import cv2
import numpy as np
import os
import time
import threading
import queue
import socket
import argparse
import signal
import sys

# Global parameters
CAPTURE_INTERVAL = 0.016  # Time between captures in seconds (target ~60 FPS)
FRAME_BUFFER_SIZE = 50    # Buffer 50 frames in memory before writing to disk
MAX_QUEUE_SIZE = 100      # Maximum frames to buffer before writing
exit_flag = False         # Global flag to signal threads to exit

def get_local_ip():
    """Get the local machine's IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = '127.0.0.1'  # Fallback to localhost
    return ip

def signal_handler(sig, frame):
    """Handle ctrl+c to ensure we write all buffered frames before exiting"""
    global exit_flag
    print("\nCapture stopped by user (Ctrl+C)")
    exit_flag = True  # Signal threads to exit

def batch_writer_worker(frame_queue, output_folder, ip_address):
    """Worker thread that handles saving frames to disk in batches"""
    frames_written = 0
    
    while True:
        try:
            # Get a batch of frames from the queue
            batch = frame_queue.get(timeout=1.0)  # Use timeout to periodically check exit flag
            
            # Exit condition
            if batch is None:
                break
            
            # Process the batch of frames
            for frame, timestamp in batch:
                # Save frame with IP + timestamp
                filename = os.path.join(output_folder, f"{ip_address.replace('.', '_')}_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                frames_written += 1
            
            # Mark batch as done
            frame_queue.task_done()
            
            print(f"Wrote batch of {len(batch)} frames to disk (total written: {frames_written})")
            
        except queue.Empty:
            # Check if we should exit
            if exit_flag and frame_queue.empty():
                break
        except Exception as e:
            print(f"Error in writer thread: {e}")
    
    print(f"Writer thread exiting after writing {frames_written} frames")

def capture_from_virtual_camera(output_folder="game_frames", interval=CAPTURE_INTERVAL, 
                               max_frames=None, capture_width=1920, capture_height=1080,
                               buffer_size=FRAME_BUFFER_SIZE):
    """Capture frames from an OBS Virtual Camera feed with optimized performance"""
    global exit_flag
    exit_flag = False  # Reset exit flag
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Get IP address and set up frame queue for batches
    ip_address = get_local_ip()
    batch_queue = queue.Queue(maxsize=10)  # Maximum number of batches in queue
    
    # Start writer threads
    num_writer_threads = min(4, os.cpu_count() or 4)
    print(f"Starting {num_writer_threads} writer threads")
    writer_threads = []
    for _ in range(num_writer_threads):
        t = threading.Thread(
            target=batch_writer_worker,
            args=(batch_queue, output_folder, ip_address),
            daemon=True
        )
        t.start()
        writer_threads.append(t)
    
    # Find OBS Virtual Camera
    camera_idx = 0
    capture = None
    
    print("Looking for OBS Virtual Camera...")
    for i in range(6):
        try:
            test_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    print(f"Found camera at index {i}")
                    answer = input(f"Is camera {i} the OBS Virtual Camera? (y/n): ")
                    if answer.lower() == 'y':
                        camera_idx = i
                        capture = test_cap
                        break
                    else:
                        test_cap.release()
            else:
                test_cap.release()
        except Exception as e:
            print(f"Error checking camera {i}: {e}")
    
    # If we haven't found the camera yet, ask for index
    if capture is None:
        try:
            camera_idx = int(input("Enter the OBS Virtual Camera index: "))
            capture = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
            if not capture.isOpened():
                print("Could not open camera at specified index")
                return
        except Exception as e:
            print(f"Error opening camera: {e}")
            return
    
    # Set buffer size and properties
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    # Set resolution
    print(f"Setting capture resolution to {capture_width}x{capture_height}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
    
    # Verify the resolution was set
    actual_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Actual capture resolution: {actual_width}x{actual_height}")
    
    # Start capturing frames
    frame_count = 0
    batch_count = 0
    start_time = time.time()
    last_capture_time = start_time
    
    # Create a local buffer to accumulate frames before sending as batch
    frame_buffer = []
    
    print(f"Starting frame capture (target: {1/interval:.1f} FPS)...")
    print(f"Using memory buffer of {buffer_size} frames before writing to disk")
    print("Press Ctrl+C to stop capturing")
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while not exit_flag:
            # Check if we've reached max frames
            if max_frames is not None and frame_count >= max_frames:
                print(f"Reached maximum frame count of {max_frames}")
                break
            
            # Calculate time until next frame should be captured
            current_time = time.time()
            elapsed_since_last = current_time - last_capture_time
            
            # Only capture if enough time has elapsed
            if elapsed_since_last >= interval:
                last_capture_time = current_time
                
                # Capture frame
                ret, frame = capture.read()
                
                if ret and frame is not None:
                    timestamp = int(current_time * 1000)  # Millisecond precision
                    
                    # Add frame to buffer
                    frame_buffer.append((frame.copy(), timestamp))
                    
                    # If buffer is full, send batch to queue
                    if len(frame_buffer) >= buffer_size:
                        try:
                            # Make a copy of the buffer to send
                            batch = frame_buffer.copy()
                            batch_queue.put(batch, block=True, timeout=1.0)
                            batch_count += 1
                            frame_buffer = []  # Clear the buffer
                            print(f"Sent batch {batch_count} ({len(batch)} frames) to disk writer")
                        except queue.Full:
                            print("Warning: Batch queue is full, waiting...")
                            # Wait briefly and retry
                            time.sleep(0.1)
                            continue
                    
                    frame_count += 1
                    
                    # Print status periodically
                    if frame_count % 100 == 0:
                        elapsed = current_time - start_time
                        capture_fps = frame_count / elapsed
                        buffer_size_now = len(frame_buffer)
                        queue_size = batch_queue.qsize()
                        print(f"Captured {frame_count} frames ({capture_fps:.2f} FPS), "
                              f"Buffer: {buffer_size_now}/{buffer_size}, Queue: {queue_size}/10 batches")
            else:
                # Small sleep to prevent CPU hogging
                time.sleep(0.001)
                
    except Exception as e:
        print(f"Error during capture: {e}")
    finally:
        # Make sure exit_flag is set
        exit_flag = True
        
        # Clean up camera
        if capture is not None:
            capture.release()
        
        # Send any remaining buffered frames to the queue
        if frame_buffer:
            print(f"Writing final batch of {len(frame_buffer)} buffered frames")
            try:
                batch_queue.put(frame_buffer, block=True, timeout=2.0)
            except queue.Full:
                print("Warning: Batch queue full, some final frames may be lost")
        
        # Wait for all batches to be written
        try:
            print(f"Waiting for {batch_queue.qsize()} remaining batches to be written...")
            batch_queue.join()
        except:
            pass
        
        # Stop writer threads
        for _ in range(len(writer_threads)):
            try:
                batch_queue.put(None, block=False)
            except:
                pass
        
        for t in writer_threads:
            t.join(timeout=2.0)
        
        # Print summary
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
            print(f"Capture complete: {frame_count} frames in {elapsed:.2f} seconds ({fps:.2f} FPS)")
            print(f"Total batches: {batch_count + (1 if frame_buffer else 0)}")

def main():
    parser = argparse.ArgumentParser(description="Capture frames from OBS Virtual Camera")
    parser.add_argument("--output", type=str, default="game_frames",
                       help="Output folder for frames")
    parser.add_argument("--interval", type=float, default=CAPTURE_INTERVAL,
                       help=f"Time between captures in seconds (default: {CAPTURE_INTERVAL})")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum number of frames to capture")
    parser.add_argument("--width", type=int, default=1920,
                       help="Capture width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                       help="Capture height (default: 1080)")
    parser.add_argument("--buffer-size", type=int, default=FRAME_BUFFER_SIZE,
                       help=f"Number of frames to buffer before writing (default: {FRAME_BUFFER_SIZE})")
    
    args = parser.parse_args()
    
    # Print setup instructions
    print("=== OBS Virtual Camera Capture ===")
    print("1. Open OBS Studio")
    print("2. Add your game as a 'Game Capture' source")
    print("3. Click 'Start Virtual Camera' in OBS")
    print("4. Once the virtual camera is running, press Enter to continue")
    input()
    
    # Use the command line resolution parameters (defaults to 1920x1080)
    capture_width = args.width
    capture_height = args.height
    print(f"Using resolution: {capture_width}x{capture_height}")
    
    capture_from_virtual_camera(
        args.output, 
        args.interval, 
        args.max_frames, 
        capture_width, 
        capture_height,
        args.buffer_size
    )

if __name__ == "__main__":
    main()