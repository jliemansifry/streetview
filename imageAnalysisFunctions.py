import cv2
from skimage.feature import hog
from scipy.misc import imread
from skimage import color
import numpy as np

def corner_frac(img, h = 400, w = 640, show = False):
    ''' 
    INPUT:  (1) 3D numpy array of image data
    OUTPUT: (1) float: the fraction of the image deemed 
                to be a corner as per the cornerHarris algorithm

    Takes a greyscale 400x640 image (default). 
    Computes the fraction of each image that is a "corner" 
    according to the cornerHarris algorithm in cv2. 
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dist_metric = dst > 0.01*1.8e8 # this 1.8e8 is a good middleground, 
                                   # else use dst.max()
    num_corners = float(len(dist_metric[dist_metric == True])
                      - len(dist_metric[dist_metric[380:][:] == True])) 
                # no corners from the google watermark ^
    if show:
        img[dst>0.01*dst.max()]=[0,0,255]
        cv2.imshow('dst',img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    corner_frac = num_corners / (h * w)
    return corner_frac

def surf(img, hessian_thresh = 5000, upright = False, show = False):
    ''' 
    INPUT:  (1) 3D numpy array of image data
            (2) integer: the cutoff to use for the hessian threshold
            (3) boolean: force features to be upright?
            (4) boolean: show the features calculated by SURF?
    
    Calculate keypoints for the input image according to SURF (Speeded-Up
    Robust Features) in cv2. The hessian threshold effectively sets 
    the number of keypoints described for each image. Fewer features will
    be calculated with a higher hessian threshold.
    '''
    surf = cv2.SURF(hessian_thresh)
    surf.upright = upright
    kp, des = surf.detectAndCompute(img, None)
    if show:
        surf_img = cv2.drawKeypoints(img, kp, 
                flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('window', surf_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return surf, kp, des

def sklearn_hog(img_name):
    ''' 
    INPUT:  (1) string: image name
    OUTPUT: (1) tuple: (1D feature vector for comparison between images, 
                3D array of image data visualizing the HOG for display)

    Sklearn's Histogram of Oriented Gradients (HOG) functionality
    was better than cv2. Here we calculate the feature vector from HOG. 
    '''
    greyscale = color.rgb2gray(imread(img_name))
    fd, hog_image = hog(greyscale, orientations=8, pixels_per_cell=(8, 8), 
                        cells_per_block=(4, 4), visualise=True, normalise = True)
    return fd, hog_image
