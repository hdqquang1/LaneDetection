import pickle
import cv2 as cv

calibration = pickle.load(open('calibration.p', 'rb'))
mtx = calibration["mtx"]
dist = calibration["dist"]

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        h,  w = frame.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # undistort
        dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
        
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imshow('dst', dst)

        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv.destroyAllWindows()
