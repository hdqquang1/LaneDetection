import cv2 as cv
import numpy as np
from numpy.linalg import inv

# Get binary image with Sobel gradient
def sobelAbsThresh(img, orient='x', ksize=3, thresh=(0, 255)):
    if orient == 'x':
        sobel = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize)
    elif orient == 'y':
        sobel = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=ksize)
    sobelAbs = np.absolute(sobel)
    sobelScaled = np.uint8(255*sobelAbs/np.max(sobelAbs))
    sobelBin = np.zeros_like(sobelScaled)
    sobelBin[(sobelScaled >= thresh[0]) & (sobelScaled <= thresh[1])] = 255

    return sobelBin


# Get binary image with magnitude of Sobel gradient
def sobelMagThresh(img, ksize=3, thresh=(0, 255)):
    sobelX = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize)
    sobelY = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=ksize)
    sobelMag = np.sqrt(sobelX*sobelX + sobelY*sobelY)
    sobelScaled = np.uint8(255*sobelMag/np.max(sobelMag))
    sobelBin = np.zeros_like(sobelScaled)
    sobelBin[(sobelScaled >= thresh[0]) & (sobelScaled <= thresh[1])] = 255

    return sobelBin


# Get binary image with direction of Sobel gradient as the artangetn of gradient
# in y divided by gradient in x
def sobelDirThresh(img, ksize=3, thresh=(0, np.pi/2)):
    sobelX = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize)
    sobelY = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=ksize)
    sobelXAbs = np.abs(sobelX)
    sobelYAbs = np.abs(sobelY)
    sobelDir = np.arctan2(sobelYAbs, sobelXAbs)
    sobelBin = np.zeros_like(sobelDir)
    sobelBin[(sobelDir >= thresh[0]) & (sobelDir <= thresh[1])] = 255

    return sobelBin


# Combine all Sobel gradient thresholding
def sobelCombinedThresh(img):
    sobelX = sobelAbsThresh(img, orient='x', ksize=21, thresh=(20, 100))
    sobelY = sobelAbsThresh(img, orient='y', ksize=21, thresh=(20, 100))
    sobelMag = sobelMagThresh(img, ksize=7, thresh=(50, 100))
    sobelDir = sobelDirThresh(img, ksize=15, thresh=(0.4, 1.3))

    sobelBin = np.zeros_like(sobelMag)
    sobelBin[((sobelX == 1) | (sobelY == 1))
             & ((sobelMag == 1) | (sobelDir == 1))] = 255

    return sobelBin


# Get binary image with HLS color and Sobel gradient thresholding
def colorHLSGradThresh(img, threshS=(100, 255),
                    threshL=(50, 255), threshSobelX=(50, 200)):
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS).astype(np.float32)
    lChannel = hls[:, :, 1]
    sChannel = hls[:, :, 2]

    # Sobel gradient in x
    sobelX = cv.Sobel(lChannel, cv.CV_64F, 1, 0)
    sobelXAbs = np.abs(sobelX)
    sobelXScaled = np.uint8(255*sobelXAbs/np.max(sobelXAbs))

    # Threshold gradient in x and color
    bin = np.zeros_like(sobelXScaled)
    bin[((sChannel >= threshS[0]) & (sChannel <= threshS[1]))
        & ((lChannel >= threshL[0]) & (lChannel <= threshL[1]))
        | ((sobelXScaled >= threshSobelX[0])
        & (sobelXScaled <= threshSobelX[1]))] = 255

    return bin


# Get binary image with BGR color and Sobel gradient thresholding
def colorBGRGradThresh(img, threshB=(150,255), threshG=(150,255), 
                       threshR=(150,255), threshSobelX=(100,200)):
    img = img.astype(np.float32)
    bChannel = img[:, :, 0]
    gChannel = img[:, :, 1]
    rChannel = img[:, :, 2]

    # Sobel gradient in x
    sobelX = cv.Sobel(bChannel, cv.CV_64F, 1, 0)
    sobelXAbs = np.abs(sobelX)
    sobelXScaled = np.uint8(255*sobelXAbs/np.max(sobelXAbs))

    # Threshold gradient in x and color
    bin = np.zeros_like(sobelXScaled)
    bin[((bChannel >= threshB[0]) & (bChannel <= threshB[1]))
        & ((gChannel >= threshG[0]) & (gChannel <= threshG[1]))
        & ((rChannel >= threshR[0]) & (rChannel <= threshR[1]))
        | ((sobelXScaled >= threshSobelX[0])
        & (sobelXScaled <= threshSobelX[1]))] = 255

    return bin

# Perspective transform
def perspectiveTrans(img, mtx, dist):
    undist = cv.undistort(img, mtx, dist, None, mtx)

    xOffset = 0
    yOffset = 0
    imgSize = (undist.shape[1], undist.shape[0])

    src = np.float32([(550, 450), (750, 450), (1250, 700), (150, 700)])
    dst = np.float32([[xOffset, yOffset], [imgSize[0]-xOffset, yOffset],
                      [imgSize[0]-xOffset, imgSize[1]-yOffset],
                      [xOffset, imgSize[1]-yOffset]])

    # Calculate perspective transform matrix from src and dst points
    M = cv.getPerspectiveTransform(src, dst)

    # Warp the image
    imgWarped = cv.warpPerspective(undist, M, imgSize)

    return imgWarped, M


# Fit lane lines
def fitLaneLines(img):
    yVals = []

    leftX = []
    rightX = []

    imgHeight = img.shape[0]
    imgWidth = img.shape[1]

    # Divide image in half vertically and use the bottom quarter to find x with
    # the most white pixels
    leftHist = np.sum(img[int(imgHeight/4):, :int(imgWidth/2)], axis=0)
    rightHist = np.sum(img[int(imgHeight/4):, int(imgWidth/2):], axis=0)

    # Estimate left and right lane's starting position
    leftStartingPeak = np.argmax(leftHist)
    leftX.append(leftStartingPeak)

    rightStartingPeak = np.argmax(rightHist)
    rightX.append(rightStartingPeak + imgWidth/2)

    # Iterate through vertical segments from bottom moving up, detecting left
    # and right lanes
    curHeight = imgHeight
    yVals.append(curHeight)
    yIncrement = 25
    colWidth = 150
    leftI = 0
    rightI = 0
    while ((curHeight - yIncrement) >= (imgHeight/4)):
        curHeight = curHeight - yIncrement

        # Center of left column
        leftColC = leftX[leftI]
        leftI += 1

        # Center of right column
        rightColC = rightX[rightI]
        rightI += 1

        # Left and right index of left column
        leftColL = max(int(leftColC - (colWidth/2)), 0)
        leftColR = min(int(leftColC + (colWidth/2)), imgWidth)

        # Left and right index of right column
        rightColL = max(int(rightColC - (colWidth/2)), 0)
        rightColR = min(int(rightColC + (colWidth/2)), imgWidth)

        leftHist = np.sum(
            img[(curHeight - yIncrement):curHeight, leftColL:leftColR], axis=0)
        rightHist = np.sum(
            img[(curHeight - yIncrement):curHeight, rightColL:rightColR], axis=0)

        leftPeak = np.argmax(leftHist)
        rightPeak = np.argmax(rightHist)

        if leftPeak:
            leftX.append(leftPeak + leftColL)
        else:
            leftX.append(leftX[leftI - 1])

        if rightPeak:
            rightX.append(rightPeak + rightColL)
        else:
            rightX.append(rightX[rightI - 1])

        yVals.append(curHeight)

    yVals = np.array(yVals)
    leftX = np.array(leftX)
    rightX = np.array(rightX)

    # Fit second order polynomial to each lane line
    leftFit = np.polyfit(yVals, leftX, 2)
    rightFit = np.polyfit(yVals, rightX, 2)

    # Calculate values for fitted lines
    leftXVals = leftFit[0]*(yVals**2) + leftFit[1]*yVals + leftFit[2]
    rightXVals = rightFit[0]*(yVals**2) + rightFit[1]*yVals + rightFit[2]

    return leftXVals, rightXVals, yVals, leftStartingPeak, rightStartingPeak


# Draw lane lines
def drawLaneLines(imgWarped, M, imgUndist, leftFitX, rightFitX, yVals):
    # Create an image to draw the lines on
    linesZero = np.zeros_like(imgWarped).astype(np.uint8)
    linesColor = np.dstack((linesZero, linesZero, linesZero))

    # Recast x and y points into usable format for cv.polylines()
    ptsLeft = np.transpose(np.vstack([leftFitX, yVals])).astype(np.int32)
    ptsRight = np.transpose(np.vstack([rightFitX, yVals])).astype(np.int32)

    # Draw lines onto the emtpy image
    cv.polylines(linesColor, [ptsLeft, ptsRight], False, (0, 0, 255), 10)
    # cv.polylines(linesColor, [ptsRight], False, (255, 255, 255), 10)

    # Warp back to original image space using inverse perspective matrix
    MInv = inv(M)
    linesUnwarp = cv.warpPerspective(linesColor, MInv,
                                     (imgUndist.shape[1], imgUndist.shape[0]))

    # Return combining the lines and the original image
    return cv.addWeighted(imgUndist, 1, linesUnwarp, 1, 0)



#  Region of interest (ROI)
def ROI(img, pts):
    mask = np.zeros_like(img)
    cv.fillPoly(mask, [pts], (255,255,255))
    imgMasked = cv.bitwise_and(img, mask)
    return imgMasked


# Average Hough Transform lines
def avgHoughLines(img):
    HoughLines = cv.HoughLinesP(img, 1, np.pi/180, 50, None, 200, 50)
    print(HoughLines)
    # lines = []

    # if HoughLines is not None:
    #     for i in range(0, len(HoughLines)):
    #         l = HoughLines[i]
            
    #         param = np.polyfit((l[0],l[1]), (l[2],l[3]), 1)
    #         slope = param[0]
    #         intercept = param[1]
    #         if (slope > 0.5):
    #             lines.append((slope, intercept))

    # avgLines = np.average(lines, axis=0)

    # return avgLines


# Draw lines:
def drawLines(img, lines):
    if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                    
                cv.line(img, (l[0], l[1]), (l[2], l[3]), 
                        (0,0,255), 3, cv.LINE_AA)


