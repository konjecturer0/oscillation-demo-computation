#! python3

from collections import deque
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import numpy as np
import threading
import queue

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

x = 122
y = 445

# Dimensions 77 x 72
width = 60
height = 180

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Output
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('detection-spring-final.avi', fourcc, 29.970662, (406, 720), True)

xs_equilibrium_queue = queue.Queue()
ys_equilibrium_queue = queue.Queue()

amplitude_y_queue = queue.Queue()

second_ys_queue = queue.Queue()

res = []

amplitude = 0

event = threading.Event()

class ThreadingEquilibrium(threading.Thread):

    def __init__(self, xs_equilibrium_queue, ys_equilibrium_queue, res):
        super().__init__()
        self.res = []
        self.xs_equilibrium_queue = xs_equilibrium_queue
        self.ys_equilibrium_queue = ys_equilibrium_queue

    def run(self):
        xs = []
        ys = []
        while True:
            xpos = self.xs_equilibrium_queue.get()
            xs.append(xpos)
            ypos = self.ys_equilibrium_queue.get()
            ys.append(ypos)
            # print("XS LIST>", xs, "YS LIST>", ys)
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            final_pos_x = (min_x + max_x) / 2
            final_pos_y = (min_y + max_y) / 2
            res.clear()
            res.append(final_pos_x)
            res.append(final_pos_y)


ys = []

class ComputeAmplitude(threading.Thread):

    def __init__(self, amplitude_y_queue, amplitude, second_ys_queue, ys):
        super().__init__()
        self.amplitude = amplitude
        self.amplitude_y_queue = amplitude_y_queue
        self.second_ys_queue = second_ys_queue

    def run(self):
        while True:
            ypos = self.second_ys_queue.get()
            ys.append(ypos)

eqThread = ThreadingEquilibrium(xs_equilibrium_queue, ys_equilibrium_queue, res)
eqThread.setDaemon(True)
eqThread.start()


amp = queue.Queue()

def get_amplitude(ys_list):
    if len(ys_list) == 0:
        pass
    else:
        min_y = min(ys_list) # 313
        equilibrium = amplitude_y_queue.get() # 415
        distancePixels = equilibrium - min_y
        amplitude = distancePixels * 2.54/72
        return amplitude
    
lock = threading.Lock()

time_elapsed = 0
op = 0
sw1 = True

# def get_timePeriod(objPos, ys_list):
#     global time_elapsed
#     global op
#     global lock
#     global sw1
#     global frameOP
#     # end_timing = time.perf_counter()
#     # print(end_timing)
#     print("POSOTIO", objPos)
#     if objPos == max(ys_list):
#         op += 1
#     elif objPos == min(ys_list):
#         op += 1
    
#     # print("~~~~~~~~~~~~~~~~~~~", op)

#     if op >= 3:
#         time_elapsed = frameOP - 216
#         sw1 = False


def displacement(amplitude, time):
    x = amplitude * np.sin(w * time)
    



frameOP = 0

# [TODO] Start looping
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

    top_left_coordinates = (x, (y - padding))

    cv2.rectangle(frame, (top_left_coordinates), (x + w, y + h - padding_bottom), (0, 255, 0), 2)


    x_pos, y_pos = top_left_coordinates

    # print("Coordinates:", "X:", x_pos, "Y:", y_pos)

    xs_equilibrium_queue.put(x_pos)
    ys_equilibrium_queue.put(y_pos)
    second_ys_queue.put(y_pos)
    


    f = tuple(res)

    if len(f) == 0:
        pass
    else:
        e_x, e_y = f
        amplitude_y_queue.put(e_y)
        # print("EQ>>", e_y)
        cv2.circle(frame, (int(e_x), int(e_y)), 3, (0, 255, 0), -1)


    ampThread = ComputeAmplitude(amplitude_y_queue, amplitude, second_ys_queue, ys)
    ampThread.setDaemon(True)
    ampThread.start()

    get_amp_Thread = threading.Thread(target=lambda q, arg1: q.put(get_amplitude(arg1)), args=(amp, ys))
    get_amp_Thread.setDaemon(True)
    get_amp_Thread.start()

    if amp.qsize() == 0:
        pass
    else:
        amplitude = amp.get() # centimeters [cm]
        # print("AMP [*]>", amplitude, "cm")
        

    # print("AMP [2]", amplitude, "[cm]")
    # Another value 8.59 for time!
    # start_time = time.perf_counter()
    # print(start_time)
    # if frameOP >= 214:
    #     if sw1:
    #         # d = str(start_time)
    #         # first_timing = d[::]
    #         # first_timing = float(first_timing)
    #         # # print(first_timing)
    #         periodThread = threading.Thread(target=get_timePeriod, args=(y_pos, ys))
    #         periodThread.setDaemon(True)
    #         periodThread.start()
            
    # print(start_time) # [TODO] ~3.4 Starting point

    # print("[+] PERIOD", time_elapsed)
    
    '''
    30 FRAMES PER SEC => 1 sec
    '''
    
    cv2.imshow("Frame", frame)
    # frameOP += 1
    # print("OP", frameOP)
    # cv2.imshow("Mask", mask)
    # out.write(frame)
    
    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break

vs.release()
# out.release()
cv2.destroyAllWindows()
