import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt

img = cv.imread('bottletrain.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
keypoints = sift.detect(gray, None)

img = cv.drawKeypoints(gray, keypoints, img)

cv.imwrite('keypoints.jpg', img)

keypoints1, descriptors1 = sift.compute(gray, keypoints)


# getting the 2nd image
img2 = cv.imread('bottletest.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
keypoints2 = sift.detect(gray2, None)
keypoints2, descriptors2 = sift.compute(gray2, keypoints2)

print(len(keypoints1), len(keypoints2))

# matching
bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

matches = sorted(matches, key=lambda x: x.distance)

img3 = cv.drawMatches(img, keypoints1, img2, keypoints2, matches[:50], img2, flags=2)

plt.imshow(img3)
plt.show()
