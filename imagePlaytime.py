import cv2
import numpy as np

test_img_list = ['data/lat_39.82753,long_-105.024_N.png', 'data/lat_39.83544,long_-102.972_N.png', 'data/lat_39.97165,long_-107.967_N.png', 'data/lat_39.90684,long_-105.007_S.png', 'data/lat_39.81343,long_-104.989_E.png', 'data/lat_39.82368,long_-104.924_N.png']
# img = cv2.imread(filename) # greyscale
def cv2_image(img_filename):
    return cv2.imread(img_filename)

def corner_frac(img, h = 400, w = 640, show = False):
    '''takes a greyscale 400x600 image (default). computes the fraction of each image that is a "corner" '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04) # ???
    # dist_metric = dst>0.01*dst.max() # dst max is gonna change! careful
    dist_metric = dst > 0.01*1.8e8 # dst max is gonna change! careful
    # i think absolute is going to be better in the long run.
    num_corners = float(len(dist_metric[dist_metric==True]) - 
            len(dist_metric[dist_metric[380:][:] == True])) # no corners 
    # from the google watermark
    # print dist_metric[dist_metric[380:][:] == True].shape # corners from 
    # watermark
    if show:
        img[dst>0.01*dst.max()]=[0,0,255]
        cv2.imshow('dst',img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    corner_frac = num_corners / (h * w)
    return corner_frac

def surf(img, show = False, hessian_thresh = 5000, upright = False):
    surf = cv2.SURF(hessian_thresh)
    surf.upright = upright
    kp, des = surf.detectAndCompute(img, None)
    
    # nOct = 4 and nOctLayer = 2 by default
    # surf.nOctaves = 10 # num of pyramid octaves keypoint detector will use
    # surf.nOctaveLayers = 10 # num images per layer in pyramid
    # surf.hessianThreshold = 5000
    # surf_img = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4) 
    if show:
        surf_img = cv2.drawKeypoints(img, kp, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('window',surf_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return surf, kp, des

def draw_detections(img, rects, thickness = 1):
    ''' straight from opencv examples. draw HOG'''
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def cv2_hog(img, show = True):
    hog = cv2.HOGDescriptor()
    # found, w = hog.detectMultiScale(img, winStride = (8,8), padding = (32,32), scale = 1.05)
    found = hog.compute(img, winStride = (8,8), padding = (32,32))
    # found_filtered = []
    # for ri, r in enumerate(found):
        # for qi, q in enumerate(found):
            # if ri != qi and inside(r, q):
                # break
        # else:
            # found_filtered.append(r)
    return found

    # h = hog.compute(img, winStride, padding, locations)
    print('%d (%d) found' % (len(found_filtered), len(found)))
    if show:
        draw_detections(img, found)
        draw_detections(img, found_filtered, 3)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return found

from PIL import Image
from skimage.feature import hog
from skimage import data, color, exposure

def sklearn_hog(img_name):
    # im = Image.open(img_name)
    # greyscale = im.convert('1')
    greyscale = color.rgb2gray(cv2_image(img_name))
    fd, hog_image = hog(greyscale, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualise=True, normalise = True)
    return fd, hog_image


def read_test_imgs():
    for im_name in test_img_list:
        im = cv2.imread(im_name)
        corner_frac(im)
        surf(im)
    
cv2.destroyAllWindows()

if __name__ == '__main__':
    img = cv2.imread(test_img_list[0])

# dist_metric = dst>0.01*dst.max()
# num_corners = float(len(dist_metric[dist_metric==True]))
# print num_corners / (400*640)
##len(dist_metric[dist_metric==False])

#img[dst>0.01*dst.max()]=[0,0,255]
#cv2.imshow('dst',img)


# def shi_corner_frac(img): # prob wont use! 
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    # corners = np.int0(corners)
    # print corners
    # for i in corners:
        # x,y = i.ravel()
        # cv2.circle(img,(x,y),3,255,-1)

