import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('/Users/rml4d7y/Personal/50from6954.jpg', 0)
img2 = cv.imread('/Users/rml4d7y/Personal/data/Screenshots/screen9654.jpg', 0)

sift = cv.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
plt.imshow(img3),plt.show()
