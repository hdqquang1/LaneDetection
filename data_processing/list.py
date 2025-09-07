from config.constants import DATASET_FOLDER
import os
import random


def get_image_paths_with_line_counts():
    """
    Analyzes the dataset to count lines in label files and
    returns a dictionary of image paths grouped by line count.
    """
    img_count = 0
    line_counts = {}

    while True:
        img_filename = f'{img_count:05}.jpg'
        label_filename = f'{img_count:05}.lines.txt'
        img_path = os.path.join(DATASET_FOLDER, img_filename)
        label_path = os.path.join(DATASET_FOLDER, label_filename)

        if not os.path.exists(img_path):
            break

        line_count = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    if line and line[0] != '#':
                        line_count += 1

        if line_count not in line_counts:
            line_counts[line_count] = []
        line_counts[line_count].append(img_path)

        img_count += 1

    print('### Line Count Analysis ###')
    for count, paths in sorted(line_counts.items()):
        print(f'Number of images with {count} line(s): {len(paths)}')
    print('---')

    return line_counts


def split_data(line_counts_dict, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits the data into training, validation, and testing sets while preserving the ratio
    of images with different line counts.
    """
    train_paths = []
    val_paths = []
    test_paths = []

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    for _, paths in line_counts_dict.items():
        random.shuffle(paths)
        total_count = len(paths)

        train_end = int(total_count * train_ratio)
        val_end = train_end + int(total_count * val_ratio)

        train_paths.extend(paths[:train_end])
        val_paths.extend(paths[train_end:val_end])
        test_paths.extend(paths[val_end:])

    return train_paths, val_paths, test_paths


def write_paths_to_file(file_path, paths):
    """Writes a list of file paths to a text file."""
    with open(file_path, 'w') as f:
        for path in paths:
            f.write(f'{path[len("dataset"):]}\n')


def main():
    # Get the image paths grouped by line count
    line_counts_dict = get_image_paths_with_line_counts()

    # Split the data
    train_paths, val_paths, test_paths = split_data(line_counts_dict)

    # Write paths to their respective files
    write_paths_to_file('dataset/list/train.txt', train_paths)
    write_paths_to_file('dataset/list/val.txt', val_paths)
    write_paths_to_file('dataset/list/test.txt', test_paths)

    print(f'Successfully wrote {len(train_paths)} paths to train.txt.')
    print(f'Successfully wrote {len(val_paths)} paths to val.txt.')
    print(f'Successfully wrote {len(test_paths)} paths to test.txt.')


if __name__ == '__main__':
    main()
