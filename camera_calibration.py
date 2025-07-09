import glob
import pickle
import cv2 as cv
import numpy as np

# Inside corners in chessboard
NX = 9
NY = 5

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Object points, like (0,0,0), (1,0,0), (2,0,0) ....
objp = np.zeros((NX*NY, 3), np.float32)
objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)

# Object points and image points arrays
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D poitns in image plane

# Read all images from path
images = glob.glob('camera_calibration/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (NX, NY), None)

    # If found, add object points and image points
    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (max(NX, NY)+2, max(NX, NY)+2),
                                   (-1, -1), criteria)
        imgpoints.append(corners)

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints,
                                                  gray.shape[::-1], None, None)

# Save camera calibration result
calibration = {}
calibration['mtx'] = mtx
calibration['dist'] = dist
pickle.dump(calibration, open('calibration.p', 'wb'))
