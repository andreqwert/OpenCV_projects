import numpy as np
import cv2

cap = cv2.VideoCapture(0)

"""Take firt frame of the video"""
ret, frame = cap.read()

"""Setup initial location of window"""
r, h, c, w = 250, 90, 400, 125
track_window = (c, r, w, h)

"""Set up the region of interest for tracking"""
roi = frame[r:r+h, c:c+w]
"""..."""
"""Конвертирует цвета в диапазоне от (0, 60, 32) до (180, 255, 255)"""
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
"""..."""
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180]) # построение гистограммы
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX) # нормировка изображения до диапазона [0, 1]

"""Setup the termination criteria, either 10 iteration or move by atlear 1 pt"""
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1) # stop the iteration if specified accuracy, epsilon, is reached

while True:
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        """Apply meanshift to get the new location"""
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        """Draw it on the image"""
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        cv2.imshow('img2', img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + '.jpg', img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()



