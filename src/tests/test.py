import cv2 as cv
import os
import numpy as np

from src.config.constants import (
    CULANE_FRAME_FOLDER,
    SCALE
)

START_FRAME = 0
STOP_FRAME = None

imgCount = START_FRAME

while True:
    imgFilename = f'{imgCount:05}.jpg'
    lineLabelFilename = f'{imgCount:05}.lines.txt'

    imgPath = os.path.join(CULANE_FRAME_FOLDER, imgFilename)
    # lineLabelPath = os.path.join(CULANE_FRAME_FOLDER, lineLabelFilename)
    lineLabelPath = os.path.join('frame_culane_backup/driver_00_01frame', lineLabelFilename)

    if not os.path.exists(imgPath):
        break
    print(f'Frame {imgCount:05}')

    frame = cv.imread(imgPath)

    # Read line label file
    with open(lineLabelPath, 'r') as f:
        for line in f.readlines():
            if (line) and (line[0] != '#'):
                # Parse all coordinates from the line
                coords = list(map(float, line.strip().split(' ')))

                # Reshape into a list of (x, y) integer tuples
                points = []
                for i in range(0, len(coords), 2):
                    points.append([int(coords[i]), int(coords[i+1])])

                points_np = np.array(points, np.int32).reshape((-1, 1, 2))

                cv.polylines(frame, [points_np], False,
                             (0, 0, 255), 5, cv.LINE_AA)

    frameHeight, frameWidth, _ = frame.shape
    frame = cv.resize(frame, (int(SCALE*frameWidth),
                      int(SCALE*frameHeight)), None)
    cv.imshow(f'frame', frame)

    key = cv.waitKey(0) & 0xFF

    if key == 27:
        break
    elif key == ord('q'):
        if STOP_FRAME is not None and imgCount >= STOP_FRAME:
            break
        imgCount += 1
        continue
    elif key == ord('w'):
        imgCount -= 1
        if imgCount < START_FRAME:
            imgCount = START_FRAME
        continue

cv.destroyAllWindows()
