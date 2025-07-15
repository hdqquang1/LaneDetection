import os
import cv2 as cv
import numpy as np
import glob
import pickle


# Focal length in mm
# F = 

# Numbers of corners in chessboard
NX = 7
NY = 5

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Initialize camera
cap = cv.VideoCapture(0)

# Directory to save frames for calibration
folderName = 'frame_calib'
if not os.path.exists(folderName):
    os.makedirs(folderName)
    print(f'Created folder {'frame_calib'}.')
imgCount = 0

print('Take frames for calibration.')

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        retChessboard, corners = cv.findChessboardCorners(gray, (NX, NY), None)

        # If found, add object points and image points
        if retChessboard:
            corners2 = cv.cornerSubPix(
                gray, corners, (max(NX, NY)+2, max(NX, NY)+2), (-1, -1), criteria)

            frameChessboard = frame.copy()
            cv.drawChessboardCorners(frameChessboard, (NX, NY),
                                     corners2, retChessboard)
            cv.imshow('frameChessboard', frameChessboard)

        cv.imshow('frame', frame)

        key = cv.waitKey(1) & 0xFF

        # Take a frame for calibration
        if key == ord('q'):
            filename = f'{imgCount:02}.jpg'
            filePath = os.path.join(folderName, filename)
            cv.imwrite(filePath, frame)
            print(f'Saved {filePath}.')
            imgCount += 1
            continue
        elif key == 27:
            break

cap.release()
cv.destroyAllWindows()

print('Camera calibration started.')
# Object points, like (0,0,0), (1,0,0), (2,0,0) ....
objp = np.zeros((NX*NY, 3), np.float32)
objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)

# Object points and image points arrays
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D poitns in image plane

imgsNames = '*.jpg'
imgs = glob.glob(os.path.join(folderName, imgsNames))

for fname in imgs:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (NX, NY), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        winSize = max(NX, NY) + 4
        corners2 = cv.cornerSubPix(
            gray, corners, (winSize, winSize), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (NX, NY), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)
print('Camera calibration finished.')

# Size of pixel in mm
fx = mtx[0][0]
fy = mtx[1][1]
# px = F / fx
# py = F / fy

# Save in pickle
calib = {}
calib['mtx'] = mtx
calib['dist'] = dist
# calib['px'] = px
# calib['py'] = py
pickle.dump(calib, open('calibration.p', 'wb'))
print('Camera intrinsic matrix and distortion saved.')
