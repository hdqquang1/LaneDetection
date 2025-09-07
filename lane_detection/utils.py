import cv2 as cv
import numpy as np
import os.path


#  Region of interest (ROI)
def ROI(img, poly):
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(img)
    for pts in poly:
        cv.fillPoly(mask, [pts], (255, 255, 255))
    imgMasked = cv.bitwise_and(img, mask)
    return imgMasked


# Draw Hough lines
def drawHoughLines(img, lines, slope_threshold=[0, 90]):
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]

            cv.line(img, (l[0], l[1]), (l[2], l[3]),
                    (0, 0, 255), 5, cv.LINE_AA)


# Annotate frame by writing points from Hough transform to a file in format of
# CULane
def annotateFrame(imgCount, frame_folder,
                  lines, slope_threshold=[0, 90], manual_mode=False):
    filename = f'{imgCount:05}.lines.txt'
    dirName = os.path.join(frame_folder, filename)
    with open(dirName, 'w+') as f:
        if lines is not None:
            for i in range(0, min(len(lines), 17)):
                l = lines[i][0]
                f.writelines(f'{l[0]} {l[1]} {l[2]} {l[3]}\n')

    f.close()


# Filter Hough transform lines through slope threshold
def filterHoughLines(lines, slope_threshold):
    if lines is None:
        return None

    l = lines[:, 0, :]

    slopesRad = np.arctan2((l[:, 1] - l[:, 3]), (l[:, 2] - l[:, 0]))
    slopesDeg = np.degrees(slopesRad)

    slopesDeg[slopesDeg < 0] += 180

    mask = (slopesDeg > slope_threshold[0]) & (slopesDeg < slope_threshold[1])

    linesFiltered = lines[mask]

    if linesFiltered.shape == 0:
        return None
    else:
        return linesFiltered


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
def SDCT(img, kernel_size=3, threshold=4):
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
    thres = threshold * np.mean(mag)

    # Binary edge map
    bin = np.uint8(mag > thres) * 255

    return bin

