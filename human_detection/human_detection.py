from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', required=True, help='path to images directory')
args = vars(ap.parse_args())


"""Initialize the HOG descriptor/person detector"""
hog = cv2.HOGDescriptor() # initializes the Histogram of Oriented Gradients descriptor
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) # to set the Support Vector Machine to be pre-trained pedestrian detector   

# loop pver the image paths
for imagePath in paths.list_images(args['images']):
    """load the image and resize it (to reduce detection time + improve detection accuracy)"""
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()

    """Проходимся окном 4 на 4 пикселя, на каждой эпохе уменьшая изображение относительно исходного пирамидально"""
    (rects, weights) = hog.detectMultiScale(image, winStride=(4,4), padding=(8,8), scale=1.05)

    """Draw the original bounding boxes"""
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 0, 255), 2)

    """
    Non-max-suppression необходимо, чтобы соединять между собой блоки обнаруженных объектов в один.
    Например, обнаружен человек, но по ошибке в прямоугольник заключены также и его голова и нога.
    Non-max-suppression приводит к такому виду, чтоб был обведён ТОЛЬКО человек целиком.

    Чем выше overlapThresh, тем больше вероятность того, что люди вдалеке НЕ будут обнаруживаться. 
    То есть по сути это - некий коэффициент 'сосредоточения'.
    """
    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65) # if probabilities are provided, sort on them instead

    """Draw the final bounding boxes"""
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    #"""Show some information on the number of bounding boxes"""
    #filename = imagePath[imagePath.rfind('/') + 1:]
    #print('[INFO] {}: {} original boxes, {} after suppression'.format(filename, len(rects), len(pick)))

    """Show the output images"""
    cv2.imshow('Before NMS', orig)
    cv2.imshow('After NMS', image)
    cv2.waitKey(0)



