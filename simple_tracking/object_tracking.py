from collections import deque # for fast appends and pops
import numpy as np
import imutils
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='path to the video file')
"""
A second optional argument, --buffer  is the maximum size of our deque, 
which maintains a list of the previous (x, y)-coordinates of the ball we are tracking.
This deque  allows us to draw the “contrail” of the ball, detailing its past locations. 
A smaller queue will lead to a shorter tail whereas a larger queue will create a longer tail 
(since more points are being tracked)
"""
ap.add_argument('-b', '--buffer', type=int, default=64, help='max buffer size') # 64 - supplied maximum buffer size
args = vars(ap.parse_args())


"""
Задаём верхнюю и нижнюю границы (в данном случае - синего) в пространстве HSV
"""
blueLower = (110, 50, 50)
blueUpper = (130, 255, 255)
pts = deque(maxlen=args['buffer'])


if not args.get('video', False): # если путь до видео НЕ указан в командной строке...
    camera = cv2.VideoCapture(0) # ...то запускаем веб-камеру
else: 
    camera = cv2.VideoCapture(args['video']) # иначе запускаем видео

# keep looping
while True:
    """Cчитываем текущий кадр"""
    (grabbed, frame) = camera.read() # grabbed indicating whether the frame was successfully read or not

    """Если последующий кадр в видео не считывается - значит, мы дошли до конца видео"""
    if args.get('video') and not grabbed:
        break

    """resize the frame, blur it and convert it to HSV color space"""
    frame = imutils.resize(frame, width=600)
    """blurred = cv2.GaussianBlur(frame, (11, 11), 0)""" # to reduce high frequency noises
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    """Строим маску для синего цвета, затем применяем расширения и рамзывания кадра 
    для того чтобы удалить мелкие пятна на маске. Исходим от минимального значения яркостей маски - на нем явно видно, за чем производится tracking"""
    mask = cv2.inRange(hsv, blueLower, blueUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    """Находим контур маски и по нему - текущий центр (x, y) круга"""
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2] # [-2] to compatibile both OpenCV 2.4 and OpenCV3.
    center = None

    """Только в том случае, если нашелся хотя бы 1 контур..."""
    if len(cnts) > 0:
        
        """find the largest contour in the mask, then use it to compute the minimum
        enclosing circle and centroid"""
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c) # help to calculate some features like center of mass of the object, area of the object etc
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) # подсчет центра фигуры

        # only proceed if the radius meets a minimum size
        if radius > 10:

            """draw the circle and centroid on the frame, then update the list of
            tracked points"""
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2) # рисует круг по данным о кадре, радиусе и координатах
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # update the points queue
    pts.appendleft(center) # append the centroid to the pts list

    # loop over the set of tracked points
    for i in range(1, len(pts)):

        """Если какая-либо из отслеживаемых точек None, игнорируем это"""
        if pts[i-1] is None or pts[i] is None:
            continue

        """otherwise, compute the thickness of the line and draw connecting lines"""
        thickness = int(np.sqrt(args['buffer'] / float(i+1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord('q'):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
