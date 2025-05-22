import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime
import win32gui
import win32con
import win32ui
from ctypes import windll

def setup_data_file():
    """Create a CSV file for storing the gaze data."""
    if not os.path.exists('recordings'):
        os.makedirs('recordings')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recordings/gaze_data_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['timestamp', 'gaze_x', 'gaze_y', 'confidence'])
    
    return filename

def window_capture(window_name):
    """Capture a specific window by name."""
    # Find the window
    hwnd = win32gui.FindWindow(None, window_name)
    
    if not hwnd:
        # If exact name not found, try partial match
        def callback(hwnd, results):
            if win32gui.IsWindowVisible(hwnd) and window_name.lower() in win32gui.GetWindowText(hwnd).lower():
                results.append(hwnd)
            return True
        
        results = []
        win32gui.EnumWindows(callback, results)
        
        if results:
            hwnd = results[0]
            print(f"Found window with partial match: '{win32gui.GetWindowText(hwnd)}'")
        else:
            return None
    
    # Get window dimensions
    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top
        
        # Ensure window is not minimized
        if width == 0 or height == 0:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.5)
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = right - left
            height = bottom - top
        
        # Capture the window using win32ui
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        
        saveDC.SelectObject(saveBitMap)
        
        # Use PrintWindow to capture the window
        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
        
        # Convert the bitmap to an OpenCV image
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        
        img = np.frombuffer(bmpstr, dtype='uint8')
        img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Clean up resources
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        return img
    
    except Exception as e:
        print(f"Error capturing window: {e}")
        return None

def main():
    print("Minimal Tobii Eye Tracking Recorder")
    print("==================================")
    
    # Ask for window name
    window_name = input("Enter window name to capture (default 'SSOverlay'): ").strip()
    if not window_name:
        window_name = "SSOverlay"

    # Set up HSV detection parameters
    # These are defaults for a white/bright overlay - adjust if needed
    h_min, s_min, v_min = 0, 0, 240  # Lower HSV bounds
    h_max, s_max, v_max = 179, 30, 255  # Upper HSV bounds
    min_area, max_area = 5, 1000  # Area thresholds

    # Allow parameter adjustment
    print("\nDefault detection settings:")
    print(f"HSV range: [{h_min},{s_min},{v_min}] to [{h_max},{s_max},{v_max}]")
    print(f"Area range: {min_area} to {max_area}")
    
    adjust = input("Do you want to adjust detection parameters? (y/n, default: n): ").strip().lower()
    if adjust == 'y':
        try:
            h_min = int(input(f"H min (0-179, default {h_min}): ") or h_min)
            s_min = int(input(f"S min (0-255, default {s_min}): ") or s_min)
            v_min = int(input(f"V min (0-255, default {v_min}): ") or v_min)
            h_max = int(input(f"H max (0-179, default {h_max}): ") or h_max)
            s_max = int(input(f"S max (0-255, default {s_max}): ") or s_max)
            v_max = int(input(f"V max (0-255, default {v_max}): ") or v_max)
            min_area = int(input(f"Min area (default {min_area}): ") or min_area)
            max_area = int(input(f"Max area (default {max_area}): ") or max_area)
        except ValueError:
            print("Invalid input, using defaults")
    
    # Choose color preset
    print("\nColor presets:")
    print("1: White (default)")
    print("2: Blue")
    print("3: Green")
    print("4: Red")
    
    preset = input("Select color preset (1-4, default 1): ").strip()
    if preset == '2':  # Blue
        h_min, s_min, v_min = 90, 100, 200
        h_max, s_max, v_max = 120, 255, 255
    elif preset == '3':  # Green
        h_min, s_min, v_min = 40, 100, 200
        h_max, s_max, v_max = 80, 255, 255
    elif preset == '4':  # Red
        h_min, s_min, v_min = 170, 100, 200
        h_max, s_max, v_max = 179, 255, 255
    
    # Setup CSV file
    filename = setup_data_file()
    
    # Create a single window for display
    cv2.namedWindow('Tobii Tracker', cv2.WINDOW_NORMAL)
    
    print("\nRecording eye tracking data to:", filename)
    print("Press 'q' to stop recording")
    
    # Open CSV file for writing
    with open(filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        try:
            frames = 0
            start_time = time.time()
            points_recorded = 0
            
            while True:
                # Capture the window
                frame = window_capture(window_name)
                
                if frame is None:
                    print("Failed to capture window. Retrying...")
                    time.sleep(0.5)
                    continue
                
                # Convert to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Create mask
                lower_bound = np.array([h_min, s_min, v_min])
                upper_bound = np.array([h_max, s_max, v_max])
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                
                # Apply morphology operations
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                mask = cv2.dilate(mask, kernel, iterations=2)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Copy for drawing
                result = frame.copy()
                
                # Process contours
                gaze_x, gaze_y = None, None
                confidence = 0
                
                if contours:
                    # Sort by area
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        
                        # Check if area is within thresholds
                        if min_area <= area <= max_area:
                            # Get center
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                # Calculate normalized coordinates
                                gaze_x = cx / frame.shape[1]
                                gaze_y = cy / frame.shape[0]
                                
                                # Calculate confidence
                                confidence = min(1.0, area / 300)
                                
                                # Draw center and area
                                cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)
                                cv2.rectangle(result, tuple(contour.min(0)[0]), tuple(contour.max(0)[0]), (0, 255, 0), 2)
                                
                                # Record data
                                timestamp = time.time()
                                csv_writer.writerow([timestamp, gaze_x, gaze_y, confidence])
                                
                                # Force flush to ensure data is written immediately
                                csvfile.flush()
                                points_recorded += 1
                                
                                # Only use the first valid contour
                                break
                
                # Add status display
                elapsed = time.time() - start_time
                frames += 1
                fps = frames / elapsed if elapsed > 0 else 0
                
                # Add recording info
                cv2.putText(result, f"Recording: {points_recorded} points", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result, f"FPS: {fps:.1f}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add coordinate display
                if gaze_x is not None:
                    text = f"Gaze: ({gaze_x:.3f}, {gaze_y:.3f})"
                    cv2.putText(result, text, (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(result, "No gaze detected", (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show only the result window
                cv2.imshow('Tobii Tracker', result)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Brief sleep to reduce CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nRecording stopped by user")
        finally:
            cv2.destroyAllWindows()
            
            # Print summary
            elapsed = time.time() - start_time
            print(f"\nRecording summary:")
            print(f"Duration: {elapsed:.1f} seconds")
            print(f"Points recorded: {points_recorded}")
            print(f"Average rate: {points_recorded/elapsed:.1f} points/second")
            print(f"Data saved to: {filename}")

if __name__ == "__main__":
    main()