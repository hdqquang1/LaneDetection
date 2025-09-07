import os
import cv2

def resize_images_in_folder(path, size=(1640, 590)):
    """
    Recursively resizes all images in a given path to a specified size using OpenCV.

    Args:
        path (str): The root directory to start searching for images.
        size (tuple): The target size (width, height) for the images.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

    for root, _, files in os.walk(path):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in valid_extensions:
                file_path = os.path.join(root, file)
                try:
                    img = cv2.imread(file_path)
                    if img is not None:
                        resized_img = cv2.resize(img, size)
                        cv2.imwrite(file_path, resized_img)
                        print(f"Resized {file_path}")
                    else:
                        print(f"Failed to read {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

if __name__ == '__main__':
    folder_path = "/home/its/Project/inpaint_outputs"
    if os.path.isdir(folder_path):
        resize_images_in_folder(folder_path)
    else:
        print("Invalid path provided.")