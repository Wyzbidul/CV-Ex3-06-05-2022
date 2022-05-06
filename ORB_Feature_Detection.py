from numpy import *
import cv2 as cv
from matplotlib.pyplot import *

img1 = cv.imread('img1.ppm')    # queryImage
img2 = cv.imread('img5.ppm')    # trainImage

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 50 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:50],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

imshow(img3)
cv.imwrite('output_ORB.ppm',img3)
cv.imwrite('jpg/output_ORB.jpg',img3)
show()