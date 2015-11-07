import cv2
import numpy as np

test_img_list = ['data/lat_39.82753,long_-105.024_N.png', 'data/lat_39.83544,long_-102.972_N.png', 'data/lat_39.97165,long_-107.967_N.png', 'data/lat_39.90684,long_-105.007_S.png', 'data/lat_39.81343,long_-104.989_E.png', 'data/lat_39.82368,long_-104.924_N.png']
# img = cv2.imread(filename) # greyscale
def cv2_image(img):
    return cv2.imread(img)

def corner_frac(img, h = 400, w = 640):
    '''takes a greyscale 400x600 image (default). computes the fraction of each image that is a "corner" '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04) # ???
    # dist_metric = dst>0.01*dst.max() # dst max is gonna change! careful
    dist_metric = dst > 0.01*1.8e8 # dst max is gonna change! careful
    # i think absolute is going to be better in the long run.
    num_corners = float(len(dist_metric[dist_metric==True]) - 
            len(dist_metric[dist_metric[380:][:] == True])) # no corners 
    corner_frac = num_corners / (h * w)
    # from the google watermark
    return corner_frac
    # print dist_metric[dist_metric[380:][:] == True].shape
    # img[dst>0.01*dst.max()]=[0,0,255]
    # cv2.imshow('dst',img)
    # cv2.waitKey(0)

def surf(img):
    surf = cv2.SURF(500)
    kp, des = surf.detectAndCompute(img, None)
    # nOct = 4 and nOctLayer = 2 by default
    # surf.nOctaves = 10 # num of pyramid octaves keypoint detector will use
    # surf.nOctaveLayers = 10 # num images per layer in pyramid
    # surf.hessianThreshold = 5000
    # surf_img = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4) 

    surf_img = cv2.drawKeypoints(img, kp, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('window',surf_img)
    cv2.waitKey(0)
    return surf, kp, des

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

