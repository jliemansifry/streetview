import numpy as np
import cv2
# http://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/

class ColorDescriptor(object):
    def __init__(self, bins):
        ''' # of bins for 3d histogram'''
        self.bins = bins

    def describe(self, image):
        '''convert the image to the HSV color space and initialize
        the features used to quantify the image '''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
 
        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        # divide the image into four rectangles/segments (top-left,
        # top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
            (0, cX, cY, h)]

        # splitting up hist into road-like segments

        # segments = [(0, 240, 0, 180), (240)]
        # top = np.array([[0,0], [640,0], [640, 220], [0, 220]])
        bottom_left = np.array([[0, 220], [320, 220], [240, 400], [0, 400]])
        center_road = np.array([[320,220], [240, 400], [400, 400]])
        bottom_right = np.array([[320,220], [640, 220], [640, 400], [400, 400]])
        segments = [bottom_left, center_road, bottom_right]
 
        # loop over the segments
        for points in segments:
            # construct a mask for each corner of the image, subtracting
            # image_quarter = np.zeros(image.shape[:2], dtype = "uint8")
            image_split = np.zeros(image.shape[:2], dtype = "uint8")
            # cv2.rectangle(image_split, (startX, startY), (endX, endY), 255, -1)
            cv2.fillPoly(image_split, [points], color = 255)
            # image_quarter = cv2.subtract(image_quarter, ellipMask)
 
            # extract a color histogram from the image, then update the
            # feature vector
            hist = self.histogram(image, image_split)
            features.extend(hist)
 
        # extract a color histogram from the elliptical region and
        # update the feature vector
        # hist = self.histogram(image, ellipMask)
        # features.extend(hist)
 
        # return the feature vector
        return features

    def histogram(self, image, mask):
        '''extract a 3D color histogram from the masked region of the
        image, using the supplied number of bins per channel; then
        normalize the histogram'''
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist).flatten()
        # return the histogram
        return hist

# image_test = cv2.imread('no_heading_images/lat_40.75376,long_-108.812no_heading.png')
# cd = ColorDescriptor((8, 12, 3))
# t = cd.describe(image_test)
