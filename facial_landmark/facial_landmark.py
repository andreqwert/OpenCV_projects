from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required=True, help='path to facial landmark predictor')
args = vars(ap.parse_args())

"""
Initialize dlib's face detector (HOG-based) and then create the
facial landmark predictor
"""
print('[INFO] loading facial landmark predictor...')
detector = dlib.get_frontal_face_detector() # HOG + linear SVM face detector
predictor = dlib.shape_predictor(args['shape_predictor'])


"""
Initialize the video stream and sleep for a bit, allowing the camera
sensor to warm up
"""
print('[INFO] camera sensor warming up...')
vs = VideoStream(src=0).start() # попытка велючить web-камеру
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    """
    Grab the frame from the threaded video stream, resize it to 
    have a maximum width of 400 pxls and covert it to grayscale
    """
    frame = vs.read()   # read the frame
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grascale frame
    rects = detector(gray, 0) # use our HOG + Linear SVM  detector to detect faces in the grayscale image

    """
    check to see if a face was detected, and if so, draw the total
    number of faces on the frame
    """
    if len(rects) > 0:
        text = '{} face(s) found'.format(len(rects))
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for rect in rects:
        # Compute the bounding box of the face and draw it on the frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)


        """
        determine the facial landmarks for the face region, then convert the facial landmark 
        (x, y)-coordinates to a numpy array
        """
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        """
        loop over the (x, y)-coordinates for the facial landmarks and draw each of them
        """
        for (i, (x, y)) in enumerate(shape):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(frame, str(i+1), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    #show the frame
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key was pressed, break from the loop
    if key == ord('q'):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
