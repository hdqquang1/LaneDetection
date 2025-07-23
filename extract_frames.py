import cv2 as cv
import os.path

from constants import (
    FRAME_FOLDER,
    YAML_PATH
)
from utils import parse_yaml

# Initialise camera
cap = cv.VideoCapture('camera_hdr.mp4')

# Get frame width and height
frameWidth = int(cap.get(3))
frameHeight = int(cap.get(4))

# Read calibration data
mtx, dist = parse_yaml(YAML_PATH)

# Refine camera matrix
newCameraMtx, _ = cv.getOptimalNewCameraMatrix(
    mtx,
    dist,
    (frameWidth, frameHeight),
    0,
    (frameWidth, frameHeight)
)

imgCount = 0

if not os.path.exists(FRAME_FOLDER):
    os.makedirs(FRAME_FOLDER)
    print(f'Created folder: {FRAME_FOLDER}.')

if not cap.isOpened():
    print('Error: Could not open video file.')

while True:
    ret, frame = cap.read()

    if ret:
        filename = f'{imgCount:05}.jpg'
        dir = os.path.join(FRAME_FOLDER, filename)
        frame = cv.undistort(frame, mtx, dist, None, newCameraMtx)
        cv.imwrite(dir, frame)
        imgCount +=1
        # print(f'Successfully write a frame at {dir}.')
    else:
        break

cap.release()
