import cv2 as cv
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import *


# Set scale factor
scale = 0.5


# Initialise camera
# cap = cv.VideoCapture('lane-detection-with-opencv/resources/project_video.mp4')
cap = cv.VideoCapture('camera_hdr_sped_up.mp4')


# Set frame width and height
frameWidth = cap.get(3)
frameHeight = cap.get(4)


while (cap.isOpened()):
    ret, frame = cap.read()
    # frame = cv.imread(
    #     './lane-detection-with-opencv/resources/test_images/test_shadow.jpg')
    # frame = cv.imread('frames/frame4500.jpg')
    frame = cv.resize(frame, (int(scale*frameWidth),
                      int(scale*frameHeight)), None)

    if ret:
        # Convert to grayscale
        frameGray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        # Read calibration
        # calib = pickle.load(open('calibration.p', 'rb'))
        # mtx = calib['mtx']
        # dist = calib['dist']

        # Undistort frame
        # frameUndist = cv.undistort(frame, mtx, dist)

        # Blur frame
        frameBlur = cv.GaussianBlur(
            frameGray, ksize=(3, 3), sigmaX=0, sigmaY=0)

        # Canny edge detector
        frameCanny = cv.Canny(frameBlur, 100, 200, None, 3)

        # Sobel gradient
        # sobelGradBin = sobelAbsThresh(frameGray, ksize=3, thresh=(20,100))
        # sobelMagBin = sobelMagThresh(frameGray, ksize=7, thresh=(50,100))
        # sobelDirBin = sobelDirThresh(frameGray, ksize=29, thresh=(1.1,1.3))
        # frameSobelBin = colorHLSGradThresh(frame)
        frameSobelBin = colorBGRGradThresh(frame)


        # Perspective transform
        # frameWarped, M = perspectiveTrans(255*frameSobelBin, mtx, dist)

        # Fit lane lines
        # leftFitX, rightFitX, yVals, _, _ = fitLaneLines(frameWarped)

        # plt.xlim(0, 1280)
        # plt.ylim(0, 720)
        # plt.plot(leftFitX, yVals, color='green', linewidth=3)
        # plt.plot(rightFitX, yVals, color='green', linewidth=3)
        # plt.gca().invert_yaxis()
        # plt.show()

        # Draw lines on frame
        # frameLines = drawLaneLines(frameWarped, M, frameUndist,
        #                         leftFitX, rightFitX, yVals)

        # ROI
        mask = np.zeros_like(frame)
        pts = np.array([[700, 100], [1100, 100], 
                        [1100, frameHeight], [0, frameHeight], [0, frameHeight*0.6]])
        ptsScaled = np.multiply(pts, scale).astype(np.int32)
        cv.fillPoly(mask, [ptsScaled], (0, 255, 0))
        frame = cv.addWeighted(frame, 1.0, mask, 0.3, 0)

        # frameROI = ROI(frameCanny, ptsScaled)
        frameROI = ROI(frameSobelBin, ptsScaled)

        # Hough Transform
        lines = cv.HoughLinesP(frameROI, 1, np.pi/180, 100, None, 50, 50)

        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]

                # Filter lines through argument/slope
                arg = np.arctan2(np.abs([l[3] - l[1]]), np.abs([l[2] - l[0]]))
                if (arg > np.pi/6):
                    cv.line(frame, (l[0], l[1]), (l[2], l[3]),
                            (0, 0, 255), 1, cv.LINE_AA)

        cv.imshow('frame', frame)
        # cv.imshow('frameCanny', frameCanny)
        # cv.imshow('frameOpening', frameOpening)
        cv.imshow('frameSobelBin', frameSobelBin)
        # cv.imshow('frameROI', frameROI)
        # # cv.imshow('frameLines', frameLines)
        # cv.imshow('frameWarped', frameWarped)
        # cv.imshow('frameUndist', frameUndist)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
