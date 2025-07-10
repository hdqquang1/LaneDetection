import pickle
import cv2 as cv

calibration = pickle.load(open('calibration.p', 'rb'))
mtx = calibration["mtx"]
dist = calibration["dist"]

img = cv.imread('camera_calibration/calibration3.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imshow('dst', dst)
cv.waitKey(0)

cv.destroyAllWindows()
