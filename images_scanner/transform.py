import numpy as np
import cv2

"""
4 point perspective transform
"""

def order_points(pts):
    # pts - a list of 4 points specifying the (x,y) coordinates at rectangle
    """
    Initialize a list of coordinates that will be ordered such that:
    - the first entry in the list is the top-left;
    - the second entry is the top-right;
    - the third entry is the bottom-right;
    - the fourth is the bottom-left.
    """

    rect = np.zeros((4, 2), dtype='float32') # 4x2 matrix of zeros

    """
    the top-left point will have the smallest sum, whereas
    the bottom-right point will have the largest sum
    """
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Returns the indices of the minimum values along an axis.
    rect[2] = pts[np.argmax(s)]

    """
    Compute the difference between the points.
    The top-right point will have the smallest difference.
    The bottom-left point will have the largest difference.
    """
    diff = np.diff(pts, axis=1) # diff out[n] = a[n+1] - a[n] on columns
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    """
    Unpack the ordered points individually
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect    #tl = top-left, tr = top-right and so on.


    """
    Compute the width of the new image. 
    It will be the maximum distance between bottom-right and bottom-left x-coords
    or the top-right and top-left x-coords
    """
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))


    """
    Compute the height of the new image. 
    It will be the maximum distance between top-right and bottom-right y-coords 
    or the top-left and bottom-left y-coords
    """
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    """
    We have the dimensions of the new image.
    Construct the set of destination points 'bird' top-down view.
    """
    dst = np.array([[0, 0],                    [maxWidth-1, 0],
                    [maxWidth-1, maxHeight-1], [0, maxHeight-1]],
                    dtype='float32')

    """
    Compute the perspective transform matrix and then apply It
    """
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight)) # warp = deformation

    return warped
