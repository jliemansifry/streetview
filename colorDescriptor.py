import numpy as np
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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

    def show_color_histogram(self, image):
        ''' Show the 2d histogram. '''
        hsv_map = np.zeros((180, 256, 3), np.uint8)
        # hsv_map = np.zeros((360, 512, 3), np.uint8) # maybe to make bigger later?
        h, s = np.indices(hsv_map.shape[:2])
        hsv_map[:,:,0] = h
        hsv_map[:,:,1] = s
        hsv_map[:,:,2] = 255
        hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)
        # cv2.imshow('hsv_map', hsv_map)
        cv2.namedWindow('hist', 0)
        hist_scale = 5
        # cv2.imshow('image', image)
        # image = cv2.pyrDown(image) # downsample image, if desired
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dark = hsv[...,2] < 32
        hsv[dark] = 0
        h = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
        # h = cv2.calcHist([hsv], [0, 1], None, [360, 512], [0, 360, 0, 512])
        h = np.clip(h*0.005*hist_scale, 0, 1)
        vis = hsv_map*h[:,:,np.newaxis] / 255.0
        cv2.imshow('hist', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def plot3d_hist(self, image, subsample = 200):
        ''' Plot a sampling of the colors of the image in HSV space (in 3d). 
        'hsv' gives the location of pixels in HSV color space, and 'rgb' 
        keeps track of the colors of these pixels for plotting
        in matplotlib. '''
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        (h, w) = image.shape[:2]
        num_pixels = h * w
        rgb = rgb.reshape(num_pixels, 3)
        hue = np.ravel(hsv[:, :, 0]).reshape(num_pixels, 1) 
        sat = np.ravel(hsv[:, :, 1]).reshape(num_pixels, 1)
        val = np.ravel(hsv[:, :, 2]).reshape(num_pixels, 1)
        
        # convert HSV values from cylindrical space (ie. theta, r, z) 
        # to x, y, z for matplotlib 3d scatter plot
        x = sat * np.cos(np.radians(hue) * 2)
        y = sat * np.sin(np.radians(hue) * 2)
        z = val
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(x[::subsample], y[::subsample], z[::subsample], color = rgb[::subsample])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); 
        ax.w_xaxis.set_ticklabels([]); ax.w_yaxis.set_ticklabels([]); ax.w_zaxis.set_ticklabels([])
        plt.show()
        
