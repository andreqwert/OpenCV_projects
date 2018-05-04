import numpy as np
import argparse
import cv2
from skimage.io import imsave


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='path to the image file')
ap.add_argument('-r', '--radius', type=int, help='radius of Gaussian blur; must be odd')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

"""При помощи размытия удаляем высокочастотные шумы и делаем фунцию minMaxLoc менее восприимчивой к шумам"""
gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0) 
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
image = orig.copy()
cv2.circle(image, maxLoc, args["radius"], (255, 0, 0), 2)
cv2.circle(image, minLoc, args["radius"], (0, 255, 0), 2)
 
# display the results of our newly improved method
cv2.imshow("Robust", image)
imsave('Robust.png', image)
cv2.waitKey(0)
