import os
from PIL import Image

def convert_png_to_jpg(path):
    """
    Recursively converts all .png images in a given path to .jpg format.
    """
    if not os.path.isdir(path):
        print("Invalid path provided.")
        return

    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.png'):
                png_path = os.path.join(root, file)
                jpg_path = os.path.splitext(png_path)[0] + '.jpg'
                
                try:
                    with Image.open(png_path) as img:
                        # Convert to RGB to handle alpha channel, as JPG does not support it
                        rgb_img = img.convert('RGB')
                        rgb_img.save(jpg_path, 'JPEG')
                        print(f"Converted '{png_path}' to '{jpg_path}'")
                except Exception as e:
                    print(f"Failed to convert {png_path}: {e}")

if __name__ == '__main__':
    folder_path = "/home/its/Project/inpaint_outputs"
    convert_png_to_jpg(folder_path)