import cv2 as cv
import os

from constants import (
    FRAME_FOLDER,
    SCALE
)

START_FRAME = 10760
STOP_FRAME = None

imgCount = START_FRAME

while True:
    imgFilename = f'{imgCount:05}.jpg'
    lineLabelFilename = f'{imgCount:05}.lines.txt'

    imgPath = os.path.join(FRAME_FOLDER, imgFilename)
    lineLabelPath = os.path.join(FRAME_FOLDER, lineLabelFilename)

    if not os.path.exists(imgPath):
        break
    print(f'Frame {imgCount:05}')

    frame = cv.imread(imgPath)

    # Read line label file
    with open(lineLabelPath, 'r') as f:
        for line in f.readlines():
            if (line) and (line[0] != '#'):
                x1, y1, x2, y2 = map(int, line.split(' '))
                cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5, cv.LINE_AA)

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
        continue

cv.destroyAllWindows()
