# src/data/data_preparation.py

import os
import shutil
import argparse
import logging
from pathlib import Path
from data_augmentation import augment_data, load_data, create_dir  # Import functions from data_augmentation.py

def copy_files(source_dir, dest_dir):
    """Copy all files and directories from source_dir to dest_dir."""
    for item in source_dir.iterdir():
        src_path = item
        dest_path = dest_dir / item.name
        try:
            if item.is_dir():
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dest_path)
        except Exception as e:
            logging.error(f"Failed to copy {src_path} to {dest_path}: {e}")

def main(args):
    # Define paths using pathlib
    project_root = Path(__file__).resolve().parents[2]  # Adjust based on directory structure
    data_dir = Path(args.data_dir).resolve() if args.data_dir else project_root / "data" / "full_dataset"
    save_dir = Path(args.save_dir).resolve() if args.save_dir else project_root / "data" / "augmented_dataset"
    augment_train = args.augment_train
    num_augments = args.num_augments

    # Define dataset splits
    splits = ['train', 'valid', 'test']

    for split in splits:
        source_split_dir = data_dir / split
        dest_split_dir = save_dir / split

        # Create destination split directory
        create_dir(dest_split_dir)

        if split == 'train' and augment_train:
            logging.info(f"Augmenting and copying training data from {source_split_dir} to {dest_split_dir}")
            # Load data
            data = load_data(source_split_dir, source_split_dir)
            if data:
                # Perform augmentation
                augment_data(data, dest_split_dir, dest_split_dir, num_augments=num_augments)
            else:
                logging.warning(f"No data found in {source_split_dir} to augment.")
        else:
            logging.info(f"Copying {split} data from {source_split_dir} to {dest_split_dir}")
            copy_files(source_split_dir, dest_split_dir)

    # Copy _darknet.labels from original train to save_dir/train if augment_train is True
    if augment_train:
        src_labels = data_dir / "train" / "_darknet.labels"
        dest_labels = save_dir / "train" / "_darknet.labels"
        if src_labels.exists():
            try:
                shutil.copy2(src_labels, dest_labels)
                logging.info(f"Copied _darknet.labels from {src_labels} to {dest_labels}")
            except Exception as e:
                logging.error(f"Failed to copy _darknet.labels: {e}")
        else:
            logging.warning(f"_darknet.labels not found in {src_labels}")

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description='Data Preparation Script for YOLOv3-tiny')

    parser.add_argument('--data_dir', type=str, default="data/full_dataset", help='Path to the full_dataset directory')
    parser.add_argument('--save_dir', type=str, default="data/augmented_dataset", help='Path to save prepared data')
    parser.add_argument('--augment_train', action='store_true', help='Flag to augment the training set')
    parser.add_argument('--num_augments', type=int, default=5, help='Number of augmentations per training image')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    main(args)
