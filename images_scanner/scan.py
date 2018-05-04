from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils   # for resizing, rotating, and cropping images

path = '/Users/User/Desktop/computer_vision/images/ex.jpg'


def load_image(path):
    """Load the image and compute the ratio of the old height to the new height, clone it and resize it"""
    
    image = cv2.imread(path)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    return image


def convert_image(image):
    """Convert image to grayscale, blur it and find edges in the image"""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # RGB to grayscale
    gray = cv2.GaussianBlur(gray, (5, 5), 0) # размытие по Гауссу с ядром размера 5х5 (?)
    edged = cv2.Canny(gray, 75, 200) # 75 и 200 - верхний и нижний порог границ детектора
    return edged


def find_contour(edged, perc=0.01):

    """
    in cv2.findContours:
    - first argument is source image, 
    - second is contour retrieval (поиск) mode, 
    - third is contour approximation method
    """
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1] #is_cv2() checks OpenCV version
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5] # ориентир на замкнутую линию из четырех ([:5]) точек

    sapproxes = np.empty(0)
    # loop over the contours
    for countour in cnts:
        
        """Вычисляем периметр (или длину кривой)"""
        peri = cv2.arcLength(countour, True) # с - кривая, True - кривая замкнута
        """Аппроксимируем полигональную кривую с заданной точностью"""
        approx = cv2.approxPolyDP(countour, perc * peri, True) # perc*peri - рез-т аппроксимации, True - кривая замкнута

        
        """If approximated contour has 4 points, then we can assume we've found our screen"""
        if len(approx) == 4:
            screenCnt = approx
            break

    try:
        """
        drawContours:
        1. image; 
        2. contours;
        3. Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
        4. Color of contour;
        5. Thickness 
        """
        print('STEP 2. Contours detection')
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 1)
        cv2.imshow('Outline', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        """Apply the 4-point transofrm to obtain a top-down view of the original image"""
        warped = four_point_transform(image, screenCnt.reshape(4, 2) * image.shape[0] / 500.0)


        """Convert the warped image to grayscale, then threshold it to give 'black and white' paper effect"""
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        T = threshold_local(warped, 11, offset=10, method='gaussian')
        warped = (warped > T).astype('uint8')*255


        """Show the original and scanned images"""
        print('STEP 3. Perspective tranform applying')
        cv2.imshow('original', imutils.resize(image, height=650))
        cv2.imshow('scanned', imutils.resize(warped, height=650))
        cv2.waitKey(0)
        
    except UnboundLocalError:
        print('Percentage is incorrect')
        perc += 0.01
        print('Reload with {}%'.format(perc*100))
        find_contour(edged, perc)


if __name__ == '__main__':
    try:
        image = load_image(path)
        edged = convert_image(image)

        print('STEP 1. Edge detection')
        cv2.imshow('image', image)
        cv2.imshow('edged', edged)
        cv2.waitKey(0) # при нажатии 'q' - переход к следующему изображению
        cv2.destroyAllWindows()

        contoured = find_contour(edged)

    except AttributeError:
        print('The path is incorrect')

