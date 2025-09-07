import os

def add_newline_to_txt_files(path):
    """
    Recursively finds all .txt files in a given directory and adds a newline
    to the end of each file.

    Args:
        path (str): The root directory to start searching for .txt files.
    """
    if not os.path.isdir(path):
        print("Invalid path. Please provide a valid directory.")
        return

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'a') as f:
                        f.write('\n')
                    print(f"Added newline to: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == '__main__':
    folder_path = "/home/its/Project/inpaint_outputs"
    add_newline_to_txt_files(folder_path)