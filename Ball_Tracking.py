'''
Created on Nov 11, 2016

@author: micro
'''
import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt
import math
from time import sleep
from time import time
from collections import deque


# "target_hypothesis" the target y value for the ball to follow
def compute_cost(target_hypothesis, current_y_pos):
    return math.ceil(((current_y_pos - target_hypothesis) ** 2) * 1000) / 1000

def compute_rolling_average(mse_vals):
    return math.ceil((sum(mse_vals) / len(mse_vals)) * 1000) / 1000

def compute_rolling_delta(current_avg, previous_avg):
    return math.ceil((current_avg - previous_avg) * 1000) / 1000

##################################################################################

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(4)

lower_green = (35, 95, 6)
upper_green = (100, 255, 255)
pts = deque(maxlen=args["buffer"])

p = 0

mse = 0
mse_vals = []

mse_avg = 0
prev_mse_avg = 0

mse_avg_delta = 0

start_time = 0
end_time = 0

while True:
    start_time = time()
    _, frame = cap.read()

    prev_mse_avg = mse_avg
    
    frame = imutils.resize(frame, width=600)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.erode(mask, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations = 2)
    
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    
    if len(contours) > 0:
        c = max(contours, key = cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] == 0:
            M["m00"] = 1
        center = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (136,255,100), 2)
            cv2.circle(frame, center, 5, (0,0,255), -1)
            pts.appendleft(center)
    
    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        
        thickness = int(np.sqrt(args["buffer"] / float(i+1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0,0,255), thickness)

    height, width = frame.shape[:2]

    target_hypothesis = math.ceil((height / 2) * 1000) / 1000

    x = math.ceil(x * 1000) / 1000
    y = math.ceil(y * 1000) / 1000

    mse = str(compute_cost(target_hypothesis, y))
    
    cv2.putText(frame, "[CENTER] (" + str(x) + ", " + str(y) + ")", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) 
                            
    mse_vals.append(compute_cost(target_hypothesis, y))
    
    cv2.line(frame, (0, int(height / 2)), (width, int(height / 2)), (255, 0, 0), 2)

    p += 1

    if p > 5:
        end_time = time()
        mse_avg = compute_rolling_average(mse_vals)
        mse_avg_delta = compute_rolling_delta(mse_avg, prev_mse_avg)
        mse_vals = []
        p = 0
        print(mse_avg_delta)

    #cv2.putText(frame, "[MSE] " + mse, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #cv2.putText(frame, "[MSE_AVG] " + str(mse_avg), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #cv2.putText(frame, "[MSE_AVG_DELTA] " + str(mse_avg_delta), (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    cv2.imshow("Frame", frame)

    #print("[INFO]  {MSE} " + str(mse) + "\n\t{MSE_AVG} " + str(mse_avg) + "\n\t{MSE_AVG_DELTA} " + str(mse_avg_delta) + "\n-------------------------------------------------")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
