import os
import random
import shutil

def split_images(src_dir, *dest_dirs):
    """
    Split files from src_dir into multiple destination directories.
    
    Args:
        src_dir (str): Source directory containing the files
        *dest_dirs (str): Variable number of destination directory paths
    """
    # Create destination directories if they don't exist
    for dest_dir in dest_dirs:
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

    # Get list of files and shuffle them
    files = os.listdir(src_dir)
    # random.shuffle(files)  # Uncomment if you want random distribution
    
    # Calculate number of files per split
    num_splits = len(dest_dirs)
    if num_splits == 0:
        print("Error: No destination directories specified")
        return
    
    files_per_split = len(files) // num_splits

    # Distribute files across destination directories
    for i, file in enumerate(files):
        src_file = os.path.join(src_dir, file)
        # Determine which destination directory this file goes to
        split_index = min(i // files_per_split, num_splits - 1)  # Ensure we don't exceed dest_dirs length
        dest_dir = dest_dirs[split_index]
        dest_file = os.path.join(dest_dir, file)
        shutil.copy(src_file, dest_file)
    
    print(f"Split {len(files)} files into {num_splits} directories")

# Example usage for 4 splits
split_images('shuffled_images_1', 
             'shuffled_images_a', 
             'shuffled_images_b', 
             'shuffled_images_c', 
             'shuffled_images_d')


# Split 7945 files into 4 directories (For shiffled_images_1)