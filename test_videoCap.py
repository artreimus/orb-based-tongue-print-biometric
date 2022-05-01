# import packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

# initialize camera and grab a reference to the raw cam capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image,
    # then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    
    # show the frame
    cv2. imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
    # if the q key was pressed, break from the loop
    if key == ord("q"):
        break
    elif key == ord("c") or key == ord("C"):
        key = int(input("Enter 1 to save, 2 to try again, 0 to exit: "))
        if key == 1:
            input = cv2.imwrite("temp.jpg", image)
            break

        elif key == 2:
            continue

        elif key == 0:
            break

cv2.destroyAllWindows()
    