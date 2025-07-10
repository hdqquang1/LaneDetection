import cv2 as cv
import os.path

imgCount = 0
folderName = 'frame'

if not os.path.exists(folderName):
    os.makedirs(folderName)
    print(f'Created folder: {folderName}.')

cap = cv.VideoCapture('camera_hdr.mp4')

if not cap.isOpened():
    print('Error: Could not open video file.')

while True:
    ret, frame = cap.read()

    if ret:
        filename = f'{imgCount:05}.jpg'
        dir = os.path.join(folderName, filename)
        cv.imwrite(dir, frame)
        imgCount +=1
        # print(f'Successfully write a frame at {dir}.')
    else:
        break

cap.release()
