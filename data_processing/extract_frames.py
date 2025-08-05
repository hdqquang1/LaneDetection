# extract_frames.py extracts and resizes frames in video. Optional argument
# outputs resized video.

from config.constants import (
    CULANE_WIDTH,
    CULANE_HEIGHT,
    DATASET_FOLDER,
    VIDEO_PATH,
    YAML_PATH
)
import argparse
import cv2 as cv
import numpy as np
import os
import yaml

# Arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video_output',
        help='path to output resized video including filename',
        default=None,
        type=str
    )
    return parser.parse_args()


# Parse yaml file to read camera matrix and distortion coefficients
def parse_yaml(yaml_path=YAML_PATH):
    with open(yaml_path, 'r') as f:
        calibration_data = yaml.safe_load(f)

    mtx = np.array(calibration_data['camera_matrix']['data']).reshape(3, 3)
    dist = np.array(calibration_data['distortion_coefficients']['data'])

    return mtx, dist


def main():
    args = get_args()

    # Initialise camera
    cap = cv.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print('Error: Could not open video file.')
        return

    # Get frame width and height
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Read calibration data
    mtx, dist = parse_yaml(YAML_PATH)

    # Refine camera matrix
    newCameraMtx, _ = cv.getOptimalNewCameraMatrix(
        mtx,
        dist,
        (frame_width, frame_height),
        0,
        (frame_width, frame_height)
    )

    img_count = 0

    os.makedirs(DATASET_FOLDER, exist_ok=True)

    out = None
    if args.video_output:
        output_dir = os.path.dirname(args.video_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv.CAP_PROP_FPS)
        out = cv.VideoWriter(args.video_output, fourcc,
                             fps, (CULANE_WIDTH, CULANE_HEIGHT))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        filename = f'{img_count:05}.jpg'
        img_path = os.path.join(DATASET_FOLDER, filename)

        # Undistort frame
        undistorted_frame = cv.undistort(
            frame, mtx, dist, None, newCameraMtx)

        # Resize frame
        resized_frame = cv.resize(
            undistorted_frame, (CULANE_WIDTH, CULANE_HEIGHT), None)

        cv.imwrite(img_path, resized_frame)
        if out:
            out.write(resized_frame)

        img_count += 1
        print(f'Successfully wrote frame {img_count} to {img_path}.')

    if out:
        print(f'Successfully wrote resized video to {args.video_output}.')
        out.release()

    cap.release()


if __name__ == '__main__':
    main()
