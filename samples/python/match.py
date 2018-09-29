import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import argparse
import imutils
import glob
import cv2



for imagePath in glob.glob("/Users/rml4d7y/Personal/data/Screenshots" + "/*.jpg"):
	# load the image, convert it to grayscale, and initialize the
	# bookkeeping variable to keep track of the matched region
    try:
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('/Users/rml4d7y/Personal/50from6954.jpg',0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.65
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

        cv2.imwrite(imagePath, image)
    except:
        pass
