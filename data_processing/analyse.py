# analyse.py provides simple dataset analysis.

from config.constants import DATASET_FOLDER
import os
from collections import Counter


def main():
    if not os.path.exists(DATASET_FOLDER):
        print(f'Error: {DATASET_FOLDER} does not exist.')
        return

    lines_counter = Counter()

    for filename in os.listdir(DATASET_FOLDER):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(DATASET_FOLDER, filename)
            with open(file_path, 'r') as f:
                num_lines = len(f.readlines())
                lines_counter[num_lines] += 1

    for count, num_img in sorted(lines_counter.items()):
        print(f'Number of images with {count} line(s): {num_img}')


if __name__ == '__main__':
    main()
