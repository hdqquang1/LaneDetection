import argparse
import cv2 as cv
import numpy as np
import os

from config.constants import (
    DATASET_FOLDER,
    YAML_PATH,
    VIDEO_PATH,
    DISPLAY_SCALE,
    CULANE_WIDTH,
    CULANE_HEIGHT,
)
from data_processing.extract_frames import parse_yaml
from lane_detection.utils import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--annotate", help="annotate the frame to file", action="store_true"
)
parser.add_argument("-n", "--no_line", help="no Hough transform", action="store_false")
parser.add_argument(
    "-m1",
    "--manual_mode_1",
    help="manual mode to label line, enter line in form of x1 y1 x2 y2",
    nargs=4,
    type=int,
    metavar=("x1", "y1", "x2", "y2"),
)
parser.add_argument(
    "-m2",
    "--manual_mode_2",
    help="manual mode to label line, enter line in form of x1 y1 x2 y2",
    nargs=4,
    type=int,
    metavar=("x1", "y1", "x2", "y2"),
)
parser.add_argument(
    "--start_frame", help="starting frame id to label", type=int, metavar="id"
)
parser.add_argument(
    "-w",
    "--warp",
    help="warp and output undistorted image based on camera calibration",
    action="store_true",
)
args = parser.parse_args()

# Initialise camera
cap = cv.VideoCapture(VIDEO_PATH)

# Get frame width and height
frameWidth = int(cap.get(3))
frameHeight = int(cap.get(4))
# frameWidth = 1640
# frameHeight = 590

# Read calibration data
mtx, dist = parse_yaml(YAML_PATH)

# Refine camera matrix
newCameraMtx, _ = cv.getOptimalNewCameraMatrix(
    mtx, dist, (frameWidth, frameHeight), 0, (frameWidth, frameHeight)
)

# Undistort image
if args.warp:
    img = cv.imread("paper/camera_calibration_raw.png")
    img = cv.undistort(img, mtx, dist, None, newCameraMtx)
    cv.imwrite("paper/camera_calibration_undistorted.png", img)
    exit()

# Threshold constants
SDCT_THRESHOLD = 3
SLOPE_THRESHOLD = [26, 60]
START_FRAME = 845
STOP_FRAME = None
MIN_LINE_LENGTH = 200
MAX_LINE_GAP = 20

# Frame counter
imgCount = START_FRAME
if args.start_frame:
    imgCount = args.start_frame

while True:
    filename = os.path.join(DATASET_FOLDER, f"{imgCount:05}.jpg")
    if not os.path.exists(filename):
        break
    frame = cv.imread(filename)
    frame = cv.undistort(frame, mtx, dist, None, newCameraMtx)
    frame = cv.resize(frame, (int(CULANE_WIDTH), int(CULANE_HEIGHT)), None)
    print(f"----------Frame {imgCount:05}----------")

    # 2-D SDCT
    frameSDCT = SDCT(frame, 7, SDCT_THRESHOLD)

    # ROI
    mask = np.zeros_like(frame)
    poly = np.array(
        [
            [
                [0, 0],
                [1240, 0],
                [1240, 590],
                # [550, frameHeight],
                [0, 590],
            ],
            # [
            #     [950, 100],
            #     [1060, 100],
            #     [1060, frameHeight],
            #     [900, frameHeight]
            # ]
        ],
        dtype=np.int32,
    )
    # for pts in poly:
    #     cv.polylines(mask, [pts], True, (0, 255, 0), 10)
    #     frame = cv.addWeighted(frame, 1.0, mask, 0.3, 0)

    frameROI = ROI(frameSDCT, poly)

    # Hough transform
    lines = None
    if args.no_line:
        lines = cv.HoughLinesP(
            frameROI, 1, np.pi / 180, 100, None, MIN_LINE_LENGTH, MAX_LINE_GAP
        )
        lines = filterHoughLines(lines, SLOPE_THRESHOLD)

    # Manually add line
    if (args.manual_mode_1) and (len(args.manual_mode_1) == 4):
        if lines is None:
            lines = np.array([[args.manual_mode_1]])
        else:
            lines = np.append(lines, [[args.manual_mode_1]], axis=0)

    if (args.manual_mode_2) and (len(args.manual_mode_2) == 4):
        if lines is None:
            lines = np.array([[args.manual_mode_2]])
        else:
            lines = np.append(lines, [[args.manual_mode_2]], axis=0)

    # Draw lines on frame
    frameHough = frame.copy()
    drawHoughLines(frameHough, lines, SLOPE_THRESHOLD)

    # Annotate frame
    if args.annotate:
        annotateFrame(imgCount, DATASET_FOLDER, lines, SLOPE_THRESHOLD)
        print(f"Annotated frame {imgCount:05}")

    # Resize frame for display
    # frameHough = cv.resize(
    #     frameHough, (int(DISPLAY_SCALE*frameWidth), int(DISPLAY_SCALE*frameHeight)), None)
    # frameROI = cv.resize(
    #     frameROI, (int(DISPLAY_SCALE*frameWidth), int(DISPLAY_SCALE*frameHeight)), None)

    # cv.imshow("frame", frame)
    # cv.imshow('frameSDCT', frameSDCT)
    cv.imshow('frameROI', frameROI)
    cv.imshow('frameHough', frameHough)

    key = cv.waitKey(0) & 0xFF

    if key == 27:
        break
    elif key == ord("q"):
        if STOP_FRAME is not None and imgCount >= STOP_FRAME:
            break
        imgCount += 1
        continue
    elif key == ord("w"):
        imgCount -= 1
        continue


cap.release()
cv.destroyAllWindows()
