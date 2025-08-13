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
from tqdm import tqdm # Import tqdm

# Arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video_output',
        help='path to output video including filename',
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
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

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

    # Initialise video writer
    out = None
    if args.video_output:
        output_dir = os.path.dirname(args.video_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv.CAP_PROP_FPS)
        out = cv.VideoWriter(args.video_output, fourcc,
                             fps, (frame_width, frame_height))
    else:
        img_count = 0
        os.makedirs(DATASET_FOLDER, exist_ok=True)

    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()

        if not ret:
            break

        # Undistort frame
        frame = cv.undistort(frame, mtx, dist, None, newCameraMtx)

        # Write to video
        if out:
            out.write(frame)
        # Output frames
        else:
            frame = cv.resize(
                frame, (CULANE_WIDTH, CULANE_HEIGHT), cv.INTER_CUBIC)
            filename = f'{img_count:05}.jpg'
            img_path = os.path.join(DATASET_FOLDER, filename)
            cv.imwrite(img_path, frame)
            img_count += 1

    if out:
        print(f'Successfully wrote video to {args.video_output}.')
        out.release()
    else:
        print(f'Successfully wrote {img_count} frames to {DATASET_FOLDER}.')

    cap.release()


if __name__ == '__main__':
    main()