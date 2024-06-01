import os
import random
import shutil
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

def move_random_files(source_dir, dest_dir, num_files=50):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for subdir, _, files in os.walk(source_dir):
        if not files:
            continue

        selected_files = random.sample(files, min(len(files), num_files))
        
        # Create the same subdirectory structure in the destination directory
        dest_subdir = subdir.replace(source_dir, dest_dir, 1)
        if not os.path.exists(dest_subdir):
            os.makedirs(dest_subdir)

        for file in selected_files:
            source_file = os.path.join(subdir, file)
            dest_file = os.path.join(dest_subdir, file)
            shutil.move(source_file, dest_file)


# Usage
# source_directory = "images/val"
# destination_directory = "dummy_images/val"
# move_random_files(source_directory, destination_directory)

def delete_except_random_files(directory, num_files=50):
    # First, we need to collect all the subdirectories
    subdirs = next(os.walk(directory))[1]
    
    # Now, we iterate over each subdirectory with a progress bar
    for subdir in tqdm(subdirs, desc="Processing subdirectories"):
        files = [f for f in listdir(join(directory, subdir)) if isfile(join(directory, subdir, f))]
        if len(files) <= num_files:
            # Skip deletion if the number of files is less than or equal to the limit
            continue

        # Randomly select files to keep
        files_to_keep = set(random.sample(files, num_files))

        # Delete files not in the list of files to keep
        for file in tqdm(files, desc=f"Processing {subdir}", leave=False):
            if file not in files_to_keep:
                os.remove(join(directory, subdir, file))
                
def delete_empty_directories(root_directory):
    for root, dirs, files in os.walk(root_directory, topdown=False):
        for directory in dirs:
            full_path = os.path.join(root, directory)
            if not os.listdir(full_path):  # Check if the directory is empty
                os.rmdir(full_path)
# Usage

