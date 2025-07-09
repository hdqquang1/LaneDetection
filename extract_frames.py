import cv2 as cv
import os.path

count = 0
folderName = 'frames'

if not os.path.exists(folderName):
    os.makedirs(folderName)
    print(f'Created folder: {folderName}.')

cap = cv.VideoCapture('camera_hdr.mp4')

if not cap.isOpened():
    print('Error: Could not open video file.')

while True:
    ret, frame = cap.read()

    if ret:
        fileName = f'frame{count}.jpg'
        dirName = os.path.join(folderName, fileName)
        cv.imwrite(dirName, frame)
        count +=1
        # print(f'Successfully write a frame at {dirName}.')
    else:
        break

cap.release()
