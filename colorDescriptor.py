import numpy as np
import cv2

class ColorDescriptor(object):
    ''' The ColorDescriptor class produces a 3d histogram of the colors in HSV
    space, either quadrant by quadrant (hist_loc == 'quadrant') or along the 
    bottom of the image with the general shape of a road. (hist_loc == 'road')
    Somewhat forked from Adrian Rosebrock at pyimagesearch.com'''

    def __init__(self, bins, hist_loc):
        ''' Initialize # of bins for 3d histogram (hue, saturation, value).
        More bins will provide a higher level of detail in describing the 
        color profile of each image. Init location to take the histogram.'''
        self.bins = bins
        self.hist_loc = hist_loc

    def describe(self, image):
        ''' Convert the image to the HSV color space and initialize
        the features used to quantify the image.'''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        if self.hist_loc == 'quadrant':
            '''
            Divide the image into quadrants: top left (A), top right (B), 
            bottom right (C), bottom left (D). 
            _________
            |_A_|_B_|
            |_D_|_C_|
            '''
            (h, w) = image.shape[:2] # img dimensions
            (cX, cY) = (int(w * 0.5), int(h * 0.5)) # center of x, y
            segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),(0, cX, cY, h)]
            for (startX, endX, startY, endY) in segments:
                image_mask = np.zeros(image.shape[:2], dtype = "uint8")
                cv2.rectangle(image_mask, (startX, startY), (endX, endY), 255, -1)
                hist = self.histogram(image, image_mask)
                features.extend(hist)
        elif self.hist_loc == 'road':
            ''' 
            Divide the image into 3 sections, ignoring the 'sky' portion of the image. 
            Bottom left (A), bottom center (B), bottom right (C). 
            B is in the shape of a road extending toward the horizon. 
            If the road is centered in the image (ie. the road is 
            oriented N, S, E, or W), this may increase the ability
            to discriminate between images (ie. dirt road or paved.)
            Set up for a 640x400 image. 
            ___________
            |         |
            |_________|
            |_A_/B\_C_|
            '''
            bottom_left = np.array([[0, 220], [320, 220], [240, 400], [0, 400]])
            center_road = np.array([[320,220], [240, 400], [400, 400]])
            bottom_right = np.array([[320,220], [640, 220], [640, 400], [400, 400]])
            segments = [bottom_left, center_road, bottom_right]
            for points in segments:
                image_mask = np.zeros(image.shape[:2], dtype = "uint8")
                cv2.fillPoly(image_mask, [points], color = 255)
                hist = self.histogram(image, image_mask)
                features.extend(hist)
        return features

    def histogram(self, image, mask):
        '''Extract a 3D color histogram from the masked region of the image
        at the given resolution of bins per channel. Normalize the histogram,
        flatten it to a 1D array, and return it.'''
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist).flatten()
        return hist
