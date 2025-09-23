import os
import pathlib

def main():
    train_list = []
    test_list = []
    img_path_str = []
    DATASET_FOLDER = '/home/its/Project/CULane_Flora'
    os.makedirs(os.path.join(DATASET_FOLDER, 'list'), exist_ok=True)

    dataset = pathlib.Path(DATASET_FOLDER)
    img_paths = sorted((list(dataset.rglob('*.jpg'))))

    # Use labels from video as ground truth
    for img_path in img_paths:
        img_path_str = '/' + str(img_path.relative_to(DATASET_FOLDER))
        # print(f"Found image: {img_path}")
        if 'driver_00'in img_path_str:
            test_list.append(img_path_str)
        elif 'Flora' in img_path_str:
            train_list.append(img_path_str)

    # Use labels from video to train and test
    # for img_path in img_paths:
    #     img_path_str.append( '/' + str(img_path.relative_to(DATASET_FOLDER)))
    # train_list, test_list = train_test_split(img_path_str, test_size=0.2, random_state=42)

    # Sort lists to ensure consistent order
    train_list = sorted(train_list)
    test_list = sorted(test_list)

    print(f"Total training images: {len(train_list)}")
    print(f"Total testing images: {len(test_list)}")
    train_list_path = os.path.join(DATASET_FOLDER, 'list', 'train.txt')
    test_list_path = os.path.join(DATASET_FOLDER, 'list', 'test.txt')

    with open(train_list_path, 'a') as f:
        for item in train_list:
            f.write(f"{item}\n")
    print(f"Train list written to {train_list_path} with {len(train_list)} entries.")

    # with open(test_list_path, 'w') as f:
    #     for item in test_list:
    #         f.write(f"{item}\n")
    # print(f"Test list written to {test_list_path} with {len(test_list)} entries.")

if __name__ == "__main__":
    main()
