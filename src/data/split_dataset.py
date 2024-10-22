import os
import shutil
import random
from typing import List, Tuple
import cv2
import numpy as np

def resize_image_and_adjust_labels(
    image_path: str,
    label_path: str,
    output_image_path: str,
    output_label_path: str,
    target_size: Tuple[int, int] = (416, 416)
) -> None:
    """
    Resize an image and adjust its corresponding YOLO labels.

    Args:
    image_path (str): Path to the input image
    label_path (str): Path to the input label file
    output_image_path (str): Path to save the resized image
    output_label_path (str): Path to save the adjusted label file
    target_size (Tuple[int, int]): Target size for the image (width, height)
    """
    # Read and resize the image
    img = cv2.imread(image_path)
    original_height, original_width = img.shape[:2]
    resized_img = cv2.resize(img, target_size)
    cv2.imwrite(output_image_path, resized_img)

    # Adjust the labels
    with open(label_path, 'r') as f:
        lines = f.readlines()

    adjusted_lines = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        
        # Adjust coordinates and dimensions
        x_center = x_center * original_width / target_size[0]
        y_center = y_center * original_height / target_size[1]
        width = width * original_width / target_size[0]
        height = height * original_height / target_size[1]

        # Ensure values are within [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        adjusted_lines.append(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")

    with open(output_label_path, 'w') as f:
        f.writelines(adjusted_lines)

def split_dataset(
    image_dir: str,
    annotation_dir: str,
    output_dir: str,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    target_size: Tuple[int, int] = (416, 416)
) -> None:
    """
    Split the dataset into train, validation, and test sets, and resize images.

    Args:
    image_dir (str): Path to the directory containing images
    annotation_dir (str): Path to the directory containing annotations
    output_dir (str): Path to the output directory
    train_split (float): Percentage of data for training (default: 0.7)
    val_split (float): Percentage of data for validation (default: 0.15)
    test_split (float): Percentage of data for testing (default: 0.15)
    target_size (Tuple[int, int]): Target size for resizing images (width, height)
    """
    # Validate split percentages
    if not abs(train_split + val_split + test_split - 1.0) < 1e-10:
        raise ValueError("Split percentages must sum to 1.")

    # Check if directories exist
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.exists(annotation_dir):
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")

    print(f"Found {len(image_files)} image files.")

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Calculate split indices
    total_images = len(image_files)
    train_end = int(total_images * train_split)
    val_end = train_end + int(total_images * val_split)

    # Split the data
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    # Create output directories
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)

    # Process and copy files to respective directories
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for file in files:
            # Process and copy image
            src_image = os.path.join(image_dir, file)
            dst_image = os.path.join(output_dir, split, 'images', file)
            
            # Process and copy annotation (assuming same name with .txt extension)
            ann_file = os.path.splitext(file)[0] + '.txt'
            src_ann = os.path.join(annotation_dir, ann_file)
            dst_ann = os.path.join(output_dir, split, 'labels', ann_file)
            
            if os.path.exists(src_image) and os.path.exists(src_ann):
                resize_image_and_adjust_labels(src_image, src_ann, dst_image, dst_ann, target_size)
                print(f"Processed and copied: {src_image} -> {dst_image}")
                print(f"Adjusted and copied: {src_ann} -> {dst_ann}")
            else:
                print(f"Warning: Image or annotation file not found: {src_image} or {src_ann}")

    # Copy classes.txt to all split directories
    classes_file = os.path.join(annotation_dir, 'classes.txt')
    if os.path.exists(classes_file):
        for split in ['train', 'val', 'test']:
            dst_classes = os.path.join(output_dir, split, 'labels', 'classes.txt')
            shutil.copy(classes_file, dst_classes)
            print(f"Copied classes.txt to {dst_classes}")
    else:
        print(f"Warning: classes.txt not found in {annotation_dir}")

    print(f"Dataset split, resize, and label adjustment complete. Output saved to {output_dir}")
    print(f"Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")

if __name__ == "__main__":
    split_dataset(
        image_dir="/home/akash/My_Projects/Optical-Plant-Identification-for-Farming-Vehicles-/data/raw_data/images",
        annotation_dir="/home/akash/My_Projects/Optical-Plant-Identification-for-Farming-Vehicles-/data/raw_data/annotations",
        output_dir="data/processed_data",
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        target_size=(416, 416)  # You can change this to (608, 608) if desired
    )