import cv2 as cv
import os
import argparse
from tqdm import tqdm


list_path = '/home/its/Project/CULane/list/test.txt'

# Train folders
# folder_to_crop = 'driver_23_30frame' # Both train and val
# folder_to_crop = 'driver_161_90frame'
# folder_to_crop = 'driver_182_30frame'

# Test folders
# folder_to_crop = 'driver_37_30frame'
# folder_to_crop = 'driver_100_30frame'
folder_to_crop = 'driver_193_90frame'

# Original resolution: 1640x590
factor = 2.3
X2 = 800
X1 = int(X2 - 164*factor)
Y1 = 275
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
                if (X1 <= x <= X2) and (Y1 <= y <= Y2):
                    lane.append([int(float((x - X1)*10/factor)), int(float((y - Y1)*10/factor))])

        if lane:
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=['write', 'display'],
        default='write',
        help="Choose 'write' to save cropped images and labels, or 'display' to show them for tuning."
    )
    args = parser.parse_args()

    with open(list_path, 'r') as f:
        img_paths = f.readlines()

    for img_path in tqdm(img_paths, desc='Processing images'):
        if folder_to_crop not in img_path:
            continue
        # Extract image and label file path
        ori_img_path = 'CULane' + img_path.strip()
        ori_label_path = ori_img_path.removesuffix('.jpg') + '.lines.txt'

        # New image and label file path
        new_img_path = 'CULane_cropped_left' + img_path.strip()
        new_label_path = new_img_path.removesuffix('.jpg') + '.lines.txt'
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)

        img = cv.imread(ori_img_path)
        lanes = read_label_file(ori_label_path)

        # Crop image
        img_cropped = img[Y1:Y2, X1:X2]
        img_cropped = cv.resize(img_cropped, (1640, 590), cv.INTER_CUBIC)
        cv.rectangle(img, (X1, Y1), (X2, Y2), (255, 0, 0), 1)

        # Draw lane points on cropped image
        for lane in lanes:
            for coord in lane:
                cv.circle(img_cropped, coord, 5, (0, 255, 0), -1)

        if args.mode == 'write':
            cv.imwrite(new_img_path, img_cropped)
            write_label_file(new_label_path, lanes)
        elif args.mode == 'display':
            cv.imshow('img', img)
            cv.imshow('img_cropped', img_cropped)
            key = cv.waitKey(0) & 0xFF
            if key == 27:  # ESC to break
                break
            elif key == ord('q'):
                continue

    return


if __name__ == '__main__':
    main()
