import numpy as np
import win32gui
import win32ui
import win32con
from PIL import Image
from time import sleep
import os
from screeninfo import get_monitors

class ScreenCapture:
    def __init__(self):
        # Get screen dimensions from primary monitor
        monitors = get_monitors()
        primary_monitor = next(m for m in monitors if m.is_primary)  # Get primary monitor
        self.w = primary_monitor.width
        self.h = primary_monitor.height
        
    def get_screenshot(self):
        # Get the desktop window handle
        hwnd = win32gui.GetDesktopWindow()
        
        # Create device contexts
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        
        # Create bitmap
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        
        # Copy screen content to bitmap
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (0, 0), win32con.SRCCOPY)
        
        # Get bitmap data
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)
        
        # Clean up
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        
        # Convert to RGB and make contiguous
        img = img[...,:3]
        img = np.ascontiguousarray(img)
        
        return img
    
    def generate_image_dataset(self):
        if not os.path.exists("images"):
            os.mkdir("images")
        while True:
            try:
                img = self.get_screenshot()
                im = Image.fromarray(img[..., [2, 1, 0]])  # Convert BGR to RGB
                im.save(f"./images/screen_img_{len(os.listdir('images'))}.jpg")
                sleep(0.3)  # Adjust this delay as needed
            except KeyboardInterrupt:
                print("Screenshot capture stopped by user")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
    
    def get_screen_size(self):
        return (self.w, self.h)

# Usage
if __name__ == "__main__":
    screencap = ScreenCapture()
    print(f"Capturing screen at resolution: {screencap.get_screen_size()}")
    screencap.generate_image_dataset()