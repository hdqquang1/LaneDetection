import cv2 as cv
import numpy as np
import os


list_path = 'CULane/list/train.txt'
factor = 2
X2 = 820
X1 = int(X2 - 164*factor)
Y1 = 300
Y2 = int(Y1 + 59*factor)


# Read label file and return labels
def read_label_file(label_path):
    # Read labels from label file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    lanes = []

    for line in lines:
        coords = line.strip('\n').split(' ')
        lane = []

        for i in range(0, len(coords), 2):
            if (i + 1) < len(coords):
                x = int(float(coords[i]))
                y = int(float(coords[i+1]))

                # Check and update coordinates based on cropped image
                if (X1 <= x <= X2) and (Y1 <= y < Y2):
                    lane.append([(x - X1)*5, (y - Y1)*5])

        lanes.append(lane)

    return lanes


# Write label file
def write_label_file(label_path, lanes):
    with open(label_path, 'w') as f:
        for lane in lanes:
            for coord in lane:
                f.write(f'{coord[0]} {coord[1]} ')
            f.write('\n')


def main():
    with open(list_path, 'r') as f:
        img_paths = f.readlines()

    test_count = 0

    for img_path in img_paths:

        # Extract image and label file path
        ori_img_path = 'CULane' + img_path.strip()
        ori_label_path = ori_img_path.removesuffix('.jpg') + '.lines.txt'

        # New image and label file path
        new_img_path = 'CULane_cropped' + img_path.strip()
        new_label_path = new_img_path.removesuffix('.jpg') + '.lines.txt'
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)

        img = cv.imread(ori_img_path)

        lanes = read_label_file(ori_label_path)

        # Crop image
        img_cropped = img[Y1:Y2, X1:X2]
        img_cropped = cv.resize(img_cropped, (1640, 590), cv.INTER_CUBIC)
        
        # Write cropped image and new label file
        cv.imwrite(new_img_path, img_cropped)
        write_label_file(new_label_path, lanes)

        # Display
        # for lane in lanes:
        #     for coord in lane:
        #         cv.circle(img_cropped, coord, 5, (0, 255, 0), -1)
        # cv.imshow('img', img)
        # cv.imshow('img_cropped', img_cropped)

        cv.waitKey(1)

    return


if __name__ == '__main__':
    main()
