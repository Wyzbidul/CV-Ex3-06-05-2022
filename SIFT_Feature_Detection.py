from numpy import *
import cv2 as cv
from matplotlib.pyplot import *

img1 = cv.imread('img1.ppm')    # queryImage
img2 = cv.imread('img5.ppm')    # trainImage

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

imshow(img3)
cv.imwrite('output_SIFT.ppm',img3)
cv.imwrite('jpg/output_SIFT.jpg',img3)

show()