import os
import shutil
import random

# Function to split data into train, test, and validation sets
def split_data(directory, split_ratio=(0.7, 0.15, 0.15)):
    train_dir = os.path.join(directory, 'train')
    test_dir = os.path.join(directory, 'test')
    val_dir = os.path.join(directory, 'validation')

    # Create directories if they don't exist
    for dir_path in [train_dir, test_dir, val_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Iterate through each .wav file
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            file_name = os.path.splitext(file)[0]
            txt_file = file_name + '.txt'
            if os.path.exists(os.path.join(directory, txt_file)):
                # Determine which directory to move the data to
                rand = random.random()
                if rand < split_ratio[0]:
                    shutil.move(os.path.join(directory, file), os.path.join(train_dir, file))
                    shutil.move(os.path.join(directory, txt_file), os.path.join(train_dir, txt_file))
                elif rand < split_ratio[0] + split_ratio[1]:
                    shutil.move(os.path.join(directory, file), os.path.join(test_dir, file))
                    shutil.move(os.path.join(directory, txt_file), os.path.join(test_dir, txt_file))
                else:
                    shutil.move(os.path.join(directory, file), os.path.join(val_dir, file))
                    shutil.move(os.path.join(directory, txt_file), os.path.join(val_dir, txt_file))

# Define the directory containing the data
data_directory = "MultiInstrument"

# Split the data into train, test, and validation sets
split_data(data_directory)
