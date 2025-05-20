import os
import shutil
import random
import argparse

def split_caltech101_dataset(original_data_dir, output_data_dir, train_split_ratio=0.8, random_seed=None):
    """
    Splits the Caltech-101 dataset into training and validation sets.

    Args:
        original_data_dir (str): Path to the original Caltech-101 dataset directory
                                 (e.g., 'path/to/101_ObjectCategories').
        output_data_dir (str): Path to the desired output directory for the split dataset
                               (e.g., 'path/to/caltech-101').
        train_split_ratio (float): The ratio of data to be used for the training set (0.0 to 1.0).
        random_seed (int, optional): Seed for the random number generator to ensure reproducibility.
    """
    if random_seed is not None:
        random.seed(random_seed)

    train_dir = os.path.join(output_data_dir, 'train')
    val_dir = os.path.join(output_data_dir, 'val')

    # Create output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Iterate through each class folder in the original dataset
    for class_name in os.listdir(original_data_dir):
        class_path = os.path.join(original_data_dir, class_name)

        # Skip if it's not a directory or if it's the 'BACKGROUND_Google' class (often excluded)
        if not os.path.isdir(class_path) or class_name == 'BACKGROUND_Google':
            continue

        print(f"Processing class: {class_name}")

        # List all image files in the class folder
        image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(image_files)

        # Calculate split index
        split_index = int(len(image_files) * train_split_ratio)

        # Split files into train and validation sets
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]

        # Create class subdirectories in train and val folders
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Copy files to the new directories
        for file_name in train_files:
            src_path = os.path.join(class_path, file_name)
            dest_path = os.path.join(train_dir, class_name, file_name)
            shutil.copy(src_path, dest_path)

        for file_name in val_files:
            src_path = os.path.join(class_path, file_name)
            dest_path = os.path.join(val_dir, class_name, file_name)
            shutil.copy(src_path, dest_path)

    print("Dataset splitting complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split Caltech-101 dataset into training and validation sets.')
    parser.add_argument('--original_data_dir', type=str, help='Path to the original Caltech-101 dataset directory (e.g., path/to/101_ObjectCategories)')
    parser.add_argument('--output_data_dir', type=str, help='Path to the desired output directory for the split dataset (e.g., path/to/caltech-101)')
    parser.add_argument('--train_split_ratio', type=float, default=0.8, help='The ratio of data to be used for the training set (0.0 to 1.0). Default is 0.8')
    parser.add_argument('--random_seed', type=int, default=None, help='Seed for the random number generator to ensure reproducibility. Default is None')

    args = parser.parse_args()

    split_caltech101_dataset(args.original_data_dir, args.output_data_dir, args.train_split_ratio, args.random_seed)