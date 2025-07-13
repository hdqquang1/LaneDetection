import cv2 as cv
import numpy as np
from numpy.linalg import inv


# Get binary image with color and Sobel gradient thresholding
def colorGradThresh(img):

    imgBGR = img.astype(np.float32)
    bChannel = imgBGR[:, :, 0]
    gChannel = imgBGR[:, :, 1]
    rChannel = imgBGR[:, :, 2]

    imgHLS = cv.cvtColor(img, cv.COLOR_BGR2HLS).astype(np.float32)
    hChannel = imgHLS[:, :, 0]
    lChannel = imgHLS[:, :, 1]
    sChannel = imgHLS[:, :, 2]

    # Sobel magnitude
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sobelX = cv.Sobel(img, cv.CV_64F, 1, 0)
    sobelY = cv.Sobel(img, cv.CV_64F, 0, 1)
    
    sobelAbsX = np.abs(sobelX)
    sobelAbsY = np.abs(sobelY)
    sobelMag = np.sqrt(np.power(sobelAbsX, 2) + np.power(sobelAbsY, 2))

    sobelScaled = np.uint8(255*sobelMag/np.max(sobelMag))

    # Thresholding
    bin = np.zeros_like(sobelScaled)
    bin[True
        & ((bChannel >= 200) & (bChannel <= 255))
        & ((gChannel >= 200) & (gChannel <= 255))
        & ((rChannel >= 200) & (rChannel <= 255))
        # & ((hChannel >= 50) & (hChannel <= 255))
        # & ((lChannel >= 100) & (lChannel <= 255))
        # & ((sChannel >= 100) & (sChannel <= 255))
        # & ((sobelScaled >= 30) & (sobelScaled <= 100))
        | ((sobelAbsX >= 100) & (sobelAbsX <= 200))] = 255

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
    cv.fillPoly(mask, [pts], (255, 255, 255))
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


# Draw Hough lines:
def drawHoughLines(img, lines, filterArg=np.pi/6):
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]

            # Filter lines through argument/slope
            arg = np.arctan2(np.abs([l[3] - l[1]]), np.abs([l[2] - l[0]]))
            if (arg > filterArg):
                cv.line(img, (l[0], l[1]), (l[2], l[3]),
                        (0, 0, 255), 1, cv.LINE_AA)
                

# Annotate frame by writing points from Hough transform to a file in format of
# CULane
def annotateFrame(dirName, lines, filterArg=np.pi/6):
    with open(dirName, 'w+') as f:
        if lines is not None:
            for i in range(0, min(len(lines), 17)):
                l = lines[i][0]

                # Filter lines through argument/slope
                arg = np.arctan2(np.abs([l[3] - l[1]]), np.abs([l[2] - l[0]]))
                if (arg > filterArg):
                    f.writelines(f'{l[0]} {l[1]} {l[2]} {l[3]}\n')

    f.close()


# Generate SDCT kernel of size N x N for bin-indices k1, k2
def SDCTkernel(N, k1, k2):
    kernel = np.zeros((N, N), dtype=np.float32)
    for m in range(N):
        for n in range(N):
            Cmk1 = np.cos(np.pi * (m + 0.5) * k1 / N)
            Cnk2 = np.cos(np.pi * (n + 0.5) * k2 / N)
            kernel[m, n] = Cmk1 * Cnk2

    return kernel

# Edge detection using SDCT-based convolution kernels
def SDCT(img, kernel_size=3):
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    N = kernel_size

    # SDCT horizontal and vertical kernels
    kernelHor = SDCTkernel(N, k1=0, k2=1)
    kernelVert = SDCTkernel(N, k1=1, k2=0)

    # Convolve with image
    F01 = cv.filter2D(img.astype(np.float32), -1, kernelHor)
    F10 = cv.filter2D(img.astype(np.float32), -1, kernelVert)

    # Magnitude of SDCT response
    mag = np.sqrt(F01**2 + F10**2)

    # Threshold 4 times mean values of convolved outputs
    thres = 4 * np.mean(mag)

    # Binary edge map
    bin = np.uint8(mag > thres) * 255

    return bin
