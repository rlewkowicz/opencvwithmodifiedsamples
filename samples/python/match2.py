import numpy as np
import cv2
import time

# local modules
from video import create_capture
from common import clock, draw_str
import threading, queue

import imutils

template = cv2.imread('/Users/rml4d7y/Personal/50from6954.jpg',0)
w, h = template.shape[::-1]
threshold = 0.8

cap = cv2.VideoCapture("udp://@127.0.0.1:2700")

q = queue.Queue(1)

def grabit(q):
    while True:
        time.sleep(.001)
        if not q.empty():
            q.get()
        ret, img = cap.read()
        q.put(img)


t = threading.Thread(target=grabit, args=(q,))
t.daemon = True
t.start()

while True:
    if not q.empty():
        img = q.get()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

        cv2.imshow('facedetect', img)

        if cv2.waitKey(5) == 27:
            break
            cv2.destroyAllWindows()
