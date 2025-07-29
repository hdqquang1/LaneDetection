import cv2 as cv
import os
from shutil import copy2

from src.config.constants import (
    CULANE_FRAME_FOLDER,
    YAML_PATH
)
from src.lane_detection.utils import parse_yaml

original_width = 1440
original_height = 928
new_width = 1640
new_height = 590

# Read calibration data
mtx, dist = parse_yaml(YAML_PATH)

# Refine camera matrix
newCameraMtx, _ = cv.getOptimalNewCameraMatrix(
    mtx,
    dist,
    (original_width, original_height),
    0,
    (original_width, original_height)
)


def resize_img(img_path, img_resized_path, new_width=new_width, new_height=new_height):
    img = cv.imread(img_path)
    img = cv.resize(img, (new_width, new_height), None)
    cv.imwrite(img_resized_path, img)
    return


def resize_line_label(line_label_path, line_label_resized_path,
                    original_width=original_width, original_height=original_height,
                    new_width=new_width, new_height=new_height):

    copy2(line_label_path, line_label_resized_path)

    updated_lines = []
    with open(line_label_resized_path, 'r') as f:
        for line in f.readlines():
            if (line) and (line[0] != '#'):
                x1, y1, x2, y2 = map(int, line.split(' '))

                x1 = new_width / original_width * x1
                x2 = new_width / original_width * x2
                y1 = new_height / original_height * y1
                y2 = new_height / original_height * y2

                updated_lines.append(f'{x1} {y1} {x2} {y2}\n')

    with open(line_label_resized_path, 'w') as f:
        f.writelines(updated_lines)
    return


def resize_video(input_video_path, output_video_path, new_width=new_width, new_height=new_height):
    cap = cv.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv.CAP_PROP_FPS)
    out = cv.VideoWriter(output_video_path, fourcc,
                         fps, (new_width, new_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        undistorted_frame = cv.undistort(frame, mtx, dist, None, newCameraMtx)
        resized_frame = cv.resize(undistorted_frame, (new_width, new_height))
        out.write(resized_frame)

    cap.release()
    out.release()
    print(f"Video resized and saved to {output_video_path}")
    return


def main():
    # img_count = 0

    # if not os.path.exists(CULANE_FRAME_FOLDER):
    #     os.makedirs(CULANE_FRAME_FOLDER)

    # while True:
    #     img_filename = f'{img_count:05}.jpg'
    #     line_label_filename = f'{img_count:05}.lines.txt'

    #     img_path = os.path.join(FRAME_FOLDER, img_filename)
    #     line_label_path = os.path.join(FRAME_FOLDER, line_label_filename)

    #     if not os.path.exists(img_path):
    #         break

    #     img_resized_path = os.path.join(CULANE_FRAME_FOLDER, img_filename)
    #     line_label_resized_path = os.path.join(
    #         CULANE_FRAME_FOLDER, line_label_filename)

    #     resize_img(img_path, img_resized_path)
    #     resize_line_label(line_label_path, line_label_resized_path)

    #     img_count += 1

    resize_video(
        '/media/hdqquang/Local/Project/camera_hdr.mp4',
        '/media/hdqquang/Local/Project/camera_hdr_resized.mp4'
    )

    return


if __name__ == '__main__':
    main()
