import numpy as np
import cv2 
from picamera2 import Picamera2

picam2 = Picamera2()

# Configure camera image types
def configCamera():
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()

def takePicture():
    # Get image and convert to BGR to HSV
    buffer = picam2.capture_array()
    color = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
    cv2.imwrite("color.jpg", color)

if __name__ == "__main__":
    configCamera()
    takePicture()
