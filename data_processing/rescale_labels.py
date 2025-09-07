import os

def rescale_coordinates(filepath, original_dims=(1440, 928), new_dims=(1640, 590)):
    """
    Rescales a single line of coordinates in a text file and overwrites the file.
    
    Args:
        filepath (str): The path to the text file.
        original_dims (tuple): The original dimensions (width, height).
        new_dims (tuple): The new dimensions (width, height).
    """
    try:
        with open(filepath, 'r') as f:
            line = f.readline().strip()

        if not line:
            print(f"Skipping empty file: {filepath}")
            return

        coords = [float(c) for c in line.split()]

        # Calculate scaling factors
        scale_x = new_dims[0] / original_dims[0]
        scale_y = new_dims[1] / original_dims[1]

        # Rescale coordinates
        rescaled_coords = []
        for i in range(0, len(coords), 2):
            x = coords[i] * scale_x
            y = coords[i+1] * scale_y
            rescaled_coords.extend([x, y])

        # Format and save the new coordinates
        new_line = " ".join([f"{c:.6f}" for c in rescaled_coords])
        with open(filepath, 'w') as f:
            f.write(new_line)
            print(f"Rescaled and updated: {filepath}")

    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except ValueError:
        print(f"Invalid coordinate data in file: {filepath}")
    except Exception as e:
        print(f"An error occurred with {filepath}: {e}")


def process_directory_recursively(root_path):
    """
    Walks through a directory recursively to find and process .txt files.
    """
    if not os.path.isdir(root_path):
        print("Invalid path. Please provide a valid directory.")
        return

    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith('.txt'):
                filepath = os.path.join(dirpath, filename)
                rescale_coordinates(filepath)

if __name__ == '__main__':
    folder_path = "/home/its/Project/inpaint_outputs"
    process_directory_recursively(folder_path)