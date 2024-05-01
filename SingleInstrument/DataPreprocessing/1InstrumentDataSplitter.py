import os
import random
import shutil

# Define the directory containing your subdirectories
main_directory = "IRMAS-TrainingData"

# List of subdirectories
directories = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

# Define the ratios for train, validation, and test data
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 1 - train_ratio - val_ratio

# Create directories for train, validation, and test data
train_dir = os.path.join(main_directory, "train")
val_dir = os.path.join(main_directory, "validation")
test_dir = os.path.join(main_directory, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Iterate through each subdirectory
for directory in directories:
    files = os.listdir(os.path.join(main_directory, directory))
    
    # Randomly shuffle the files
    random.shuffle(files)
    
    # Calculate the split indices
    train_index = int(train_ratio * len(files))
    val_index = int((train_ratio + val_ratio) * len(files))
    
    # Split the files into train, validation, and test sets
    train_files = files[:train_index]
    val_files = files[train_index:val_index]
    test_files = files[val_index:]
    
    # Copy files to train directory
    for file in train_files:
        src = os.path.join(main_directory, directory, file)
        dst = os.path.join(train_dir, directory, file)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    
    # Copy files to validation directory
    for file in val_files:
        src = os.path.join(main_directory, directory, file)
        dst = os.path.join(val_dir, directory, file)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    
    # Copy files to test directory
    for file in test_files:
        src = os.path.join(main_directory, directory, file)
        dst = os.path.join(test_dir, directory, file)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

print("Data split into train, validation, and test sets successfully.")
