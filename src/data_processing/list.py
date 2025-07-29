import os
import random

from src.config.constants import (
    CULANE_FRAME_FOLDER
)


def createImgLists(base_folder, output_folder, train_ratio, val_ratio, test_ratio, img_folder='driver_00_01frame'):
    """
    Creates train, validation, and test lists of image file paths.

    Args:
        base_folder (str): The base directory where the image folder and list folder are located.
        output_folder (str): The name of the folder where the train/val/test lists will be saved.
        train_ratio (int): The desired count for the training set.
        val_ratio (int): The desired count for the validation set.
        test_ratio (int): The desired count for the test set.
        img_folder (str): The name of the folder containing the images.
    """
    listDir = os.path.join(base_folder, output_folder)
    allPath = os.path.join(listDir, 'all.txt')
    trainPath = os.path.join(listDir, 'train.txt')
    valPath = os.path.join(listDir, 'val.txt')
    testPath = os.path.join(listDir, 'test.txt')

    os.makedirs(listDir, exist_ok=True)

    imgPath = os.path.join(base_folder, img_folder)
    imgFiles = []

    if os.path.isdir(imgPath):
        for filename in os.listdir(imgPath):
            if filename.lower().endswith('.jpg'):
                imgFiles.append(f'/{img_folder}/{filename}\n')
    else:
        print(f"Error: Image folder '{imgPath}' not found.")
        return

    random.shuffle(imgFiles)

    totalImg = len(imgFiles)
    totalRatio = train_ratio + val_ratio + test_ratio

    trainCount = int(totalImg * (train_ratio / totalRatio))
    valCount = int(totalImg * (val_ratio / totalRatio))
    testCount = totalImg - trainCount - valCount

    trainFiles = imgFiles[:trainCount]
    valFiles = imgFiles[trainCount: trainCount + valCount]
    testFiles = imgFiles[trainCount + valCount:]

    def sort_key(filepath):
        return int(os.path.basename(filepath).split('.')[0])

    imgFiles.sort(key=sort_key)
    trainFiles.sort(key=sort_key)
    valFiles.sort(key=sort_key)
    testFiles.sort(key=sort_key)

    with open(allPath, 'w') as fTrain:
        fTrain.writelines(imgFiles)

    # with open(trainPath, 'w') as fTrain:
    #     fTrain.writelines(trainFiles)

    # with open(valPath, 'w') as fVal:
    #     fVal.writelines(valFiles)

    # with open(testPath, 'w') as fTest:
    #     fTest.writelines(testFiles)

    print(f"Generated image lists in: {listDir}")
    print(f"Train files written: {len(trainFiles)}")
    print(f"Validation files written: {len(valFiles)}")
    print(f"Test files written: {len(testFiles)}")
    print(f"Total images actually processed (from imgFiles list): {totalImg}")


if __name__ == "__main__":
    createImgLists(
        base_folder='frame_culane_backup',
        output_folder='list',
        train_ratio=88880,
        val_ratio=9675,
        test_ratio=34680,
        img_folder='driver_00_01frame'
    )
