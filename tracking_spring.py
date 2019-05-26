#! python3

from collections import deque
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import numpy as np

'''
Step #1: Detect the presence of a colored ball using computer vision techniques.
Step #2: Track the ball as it moves around in the video frames, drawing its previous positions as it moves.
'''


def nothing(x):
    pass

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing) # Lower Hue
cv2.createTrackbar("LS", "Tracking", 49, 255, nothing) # Lower Saturation
cv2.createTrackbar("LV", "Tracking", 42, 255, nothing) # Lower Value
cv2.createTrackbar("UH", "Tracking", 63, 255, nothing) # Upper Hue
cv2.createTrackbar("US", "Tracking", 107, 255, nothing) # Upper Saturation    
cv2.createTrackbar("UV", "Tracking", 150, 255, nothing) # Upper Value


'''

Array Color Data:

#1 - 0,16,26(L) 49,114,127(U)
#2 - 2,46,60(L) 72,91,99(U)
#3 - 0,46,86(L) 72,186,125(U) # not good
#4 - 0,46,21(L) 21,106,111(U) # pretty good
#5 - 0,70,19(L) 21,106,111(U) # seems perfect
#6 - 0,70,19(L) 21,107,215(U) # good
#7 - 0,55,72(L) 56,111,212(u) # perfect for implementation {found}

'''
# l_b = np.array([0, 16, 26])
# u_b = np.array([40, 114, 110])

vs = cv2.VideoCapture('video\\spring-system.mp4')

time.sleep(2.0)


x = 122
y = 445

# Dimensions 77 x 72
width = 60
height = 180

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Output
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('detection-spring-final.avi', fourcc, 29.970662, (406, 720), True)


# Start looping
while True:
    
    _, frame = vs.read()

    if frame is None:
        break

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    
    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    _, track_window = cv2.meanShift(mask, (x, y, width, height), term_criteria)
    x, y, w, h = track_window
    padding = 30
    padding_bottom = 35
    cv2.rectangle(frame, (x, (y - padding)), (x + w, y + h - padding_bottom), (0, 255, 0), 2)



    # print("Coordinates:", "X:", x, "Y:", y)
    print("Coordinates:", "W:", w, "H:", h)

    # if y 

    
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    # out.write(frame)
    
    key = cv2.waitKey(60) & 0xFF

    if key == ord('q'):
        break


def displacement(time):
    x = amplitude * np.sin(w * time)

vs.release()
# out.release()
cv2.destroyAllWindows()
