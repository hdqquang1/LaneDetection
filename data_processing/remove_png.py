import os

def remove_png_files_in_folder(path):
    """
    Recursively removes all .png files from a given directory.
    """
    if not os.path.isdir(path):
        print(f"Error: The path '{path}' is not a valid directory.")
        return

    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Successfully removed: {file_path}")
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while removing {file_path}: {e}")

if __name__ == '__main__':
    folder_path = "/home/its/Project/inpaint_outputs"
    remove_png_files_in_folder(folder_path)