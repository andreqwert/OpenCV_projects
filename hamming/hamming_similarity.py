from skimage.io import imread
import cv2
import matplotlib.pyplot as plt
import numpy as np


face_path0 = "/Users/user/Desktop/faces/face10.jpg"
face_path2 = "/Users/user/Desktop/faces/face11.jpg"


def translate_pxls_to_bits(img, mean):
    img = img.ravel()
    img[img <= mean] = 0
    img[img > mean] = 1
    return img


def compute_distances(face_path):

    img = imread(face_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = int(np.mean(img.ravel()))
    img_in_bits = [str(bit) for bit in translate_pxls_to_bits(img, mean)]

    s = img_in_bits[0]
    for bit in img_in_bits:
        s += bit
    #s = hash(int(s))
    #print(s)
    return s


def compute_similarity():
    img1 = compute_distances(face_path0)
    img2 = compute_distances(face_path2)
    img1, img2 = str(img1), str(img2)

    assert len(img1) != len(img2), 'Images lengths are different'
    hamming_s = sum(i != j for i, j in zip(img1, img2))

    same_bits = 0
    for bit in zip(img1, img2):
        if bit[0] == bit[1]:
            same_bits += 1
        else:
            pass
    similarity_quality = (same_bits / len(img1)) * 100
    print('Similarity_quality = {}%'.format(similarity_quality))
    return similarity_quality



compute_similarity()



