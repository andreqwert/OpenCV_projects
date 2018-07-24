# coding: utf-8
import os
import cv2
import dlib
import numpy as np
import argparse
import time
import mxnet as mx
from mtcnn_detector import MtcnnDetector
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.18-4.06.hdf5"
modhash = '89f56a39a78454e96379348bddd78c0d'

directory_to_write = '/Users/user/Desktop/dima_init/faces'
camera = cv2.VideoCapture("/Users/user/Desktop/video2.mp4")

detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker=1, accurate_landmark = False)


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None, help="path to weight file (e.g. weights.18-4.06.hdf5)")
    parser.add_argument("--depth", type=int, default=16, help="depth of network")
    parser.add_argument("--width", type=int, default=8, help="width of network")
    args = parser.parse_args()
    return args


def crop_face(face, img, img_size=64):
    """Resize and reshape faces to give it to predictor"""

    input_img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (64,64))
    img_h, img_w, _ = np.shape(input_img)
    cropped_face = np.empty((1, img_size, img_size, 3))
    cropped_face[0, :, :, :] = cv2.resize(face[:, :, :], (img_size, img_size))
    return cropped_face


def process_results(results):
    """Results processing to get the predicted age and gender only"""

    predicted_genders = results[0]
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_age = int((results[1].dot(ages).flatten())[0])
    if predicted_genders[0][0] > 0.5:
        gender = "F"
    else:
        gender = "M"
    return predicted_age, gender


def main(img_size=64):
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file
    if not weight_file:
        weight_file = get_file("weights.18-4.06.hdf5", pretrained_model, cache_subdir="pretrained_models",
                                file_hash=modhash, cache_dir=os.path.dirname(os.path.abspath(__file__)))

    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)
    prev_center = (0, 0)
    n = 0

    while True:
        grab, frame = camera.read()
        frame = cv2.resize(frame, (360, 640))
        img = cv2.resize(frame, (180, 320))

        t1 = time.time()
        results = detector.detect_face(img)
        print 'time: ', time.time() - t1

        if not results is None:
            total_boxes = results[0]
            points = results[1]

            draw = frame.copy()
            for b in total_boxes:
                center = ((b[0]+b[2])/2, (b[1]+b[3])/2)
                if (np.abs(center[1] - 140) < 5):
                    if (center[1] > prev_center[1] and np.sqrt((prev_center[0]-center[0])*(prev_center[0]-center[0]) + (prev_center[1]-center[1])*(prev_center[1]-center[1])) < 12): continue
                    prev_center = center

                    face = frame[2*int(b[1])-10:2*int(b[3])+10, 2*int(b[0])-10:2*int(b[2])+10]
                    write_face = cv2.imwrite(directory_to_write + 'face' + str(n) + '.jpg', face)
                    if (write_face):
                        n += 1
                        cv2.rectangle(draw, (2*int(b[0]), 2*int(b[1])), (2*int(b[2]), 2*int(b[3])), (0, 255, 0), 3) # to make rectangles around the face

                        cropped_face = crop_face(face, img) # for further processing we pick out the faces only.
                        results = model.predict(cropped_face)
                        age, gender = process_results(results)
                        cv2.putText(draw, '{}, {}'.format(age, gender), (int(b[1])-10, int(b[3])+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            draw = frame
        cv2.imshow("detection result", draw)
        cv2.waitKey(1)
    camera.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
