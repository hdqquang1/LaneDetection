# test.py provides manual control through frames to check for labelling errors.

from config.constants import (
    DISPLAY_SCALE
)
import cv2 as cv
import numpy as np
import os

DATASET_FOLDER = '/home/its/Project/inpaint_outputs' # Change to your dataset folder
# Adjust start and stop frame as needed
START_FRAME = 0
STOP_FRAME = None


def main():
    img_count = START_FRAME

    while True:
        img_filename = f'{img_count:05}.jpg'
        label_filename = f'{img_count:05}.lines.txt'

        img_path = os.path.join(DATASET_FOLDER, img_filename)
        label_path = os.path.join(DATASET_FOLDER, label_filename)

        if not os.path.exists(img_path):
            print(f'Error: {img_path} does not exist.')
            break

        print(f'Displaying frame: {img_path}')

        img = cv.imread(img_path)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    if line and line[0] != '#':
                        coords = list(map(float, line.strip().split(' ')))
                        points = [
                            [int(coords[i]), int(coords[i+1])]
                            for i in range(0, len(coords), 2)
                        ]
                        points_np = np.array(
                            points, np.int32).reshape((-1, 1, 2))
                        cv.polylines(img, [points_np], False,
                                     (0, 0, 255), 5, cv.LINE_AA)
        else:
            print(f'Warning: {label_path} does not exist.')

        img_height, img_width, _ = img.shape
        resized_img = cv.resize(img, (int(DISPLAY_SCALE * img_width),
                                int(DISPLAY_SCALE * img_height)),
                                interpolation=cv.INTER_LINEAR)
        cv.imshow('img', resized_img)

        key = cv.waitKey(0) & 0xFF

        # ESC to quit, q to play forward,  w to play backward
        if key == 27:  # ESC key
            break
        elif key == ord('q'):
            if STOP_FRAME is not None and img_count >= STOP_FRAME:
                break
            img_count += 1
        elif key == ord('w'):
            img_count -= 1
            if img_count < START_FRAME:
                img_count = START_FRAME

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
