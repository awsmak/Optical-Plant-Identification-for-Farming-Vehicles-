import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from pathlib import Path

def augment_dataset(
    input_dir: str,
    output_dir: str,
    num_augmentations: int = 5
) -> None:
    """
    Augment the dataset and save the augmented data.

    Args:
    input_dir (str): Path to the input directory containing 'images' and 'labels' subdirectories
    output_dir (str): Path to the output directory where augmented data will be saved
    num_augmentations (int): Number of augmentations to create for each image (default: 5)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_img_dir = input_dir / "images"
    input_label_dir = input_dir / "labels"
    output_img_dir = output_dir / "images"
    output_label_dir = output_dir / "labels"

    # Check if input directories exist
    if not input_img_dir.exists():
        raise FileNotFoundError(f"Input image directory not found: {input_img_dir}")
    if not input_label_dir.exists():
        raise FileNotFoundError(f"Input label directory not found: {input_label_dir}")
    
    # Find image files with various extensions (case-insensitive)
    image_files = list(input_img_dir.glob("*.[jJ][pP][gG]")) + \
                  list(input_img_dir.glob("*.[jJ][pP][eE][gG]")) + \
                  list(input_img_dir.glob("*.[pP][nN][gG]"))
    
    if not image_files:
        raise ValueError(f"No image files (jpg, jpeg, png) found in {input_img_dir}")

    print(f"Found {len(image_files)} image files.")

    # Create output directories if they don't exist
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # Copy classes.txt to the output label directory
    classes_file = input_label_dir / "classes.txt"
    if classes_file.exists():
        with open(classes_file, "r") as f:
            classes = f.read().strip().split("\n")
        with open(output_label_dir / "classes.txt", "w") as f:
            f.write("\n".join(classes))
    else:
        print(f"Warning: classes.txt not found in {input_label_dir}")

    # Define augmentation pipeline
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.GaussNoise(p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # Process all images in the input directory
    for image_path in tqdm(image_files, desc="Augmenting images"):
        label_path = input_label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            # Read image and labels
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            with open(label_path, "r") as f:
                lines = f.read().strip().split("\n")
            
            bboxes = []
            class_labels = []
            for line in lines:
                try:
                    class_id, *bbox = map(float, line.split())
                    bboxes.append(bbox)
                    class_labels.append(int(class_id))
                except ValueError:
                    print(f"Warning: Invalid bounding box in {label_path}. Skipping this box.")
                    continue
            
            if not bboxes:
                print(f"Warning: No valid bounding boxes found in {label_path}. Skipping this image.")
                continue

            # Augment and save
            for i in range(num_augmentations):
                try:
                    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_class_labels = augmented['class_labels']
                except ValueError as e:
                    print(f"Warning: Error during augmentation for {image_path}. Skipping this augmentation.")
                    print(f"Error details: {str(e)}")
                    continue
                
                # Save augmented image
                aug_image_filename = f"{image_path.stem}_aug_{i}{image_path.suffix}"
                cv2.imwrite(str(output_img_dir / aug_image_filename), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                
                # Save augmented labels
                aug_label_filename = f"{image_path.stem}_aug_{i}.txt"
                with open(output_label_dir / aug_label_filename, "w") as f:
                    for bbox, class_id in zip(aug_bboxes, aug_class_labels):
                        f.write(f"{class_id} {' '.join(map(str, bbox))}\n")
        else:
            print(f"Warning: Label file not found for {image_path.name}")

    print(f"Augmentation complete. Augmented data saved in: {output_dir}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total augmented images created: {len(image_files) * num_augmentations}")

if __name__ == "__main__":
    augment_dataset(
        input_dir="data/processed_data/train",
        output_dir="data/processed_data/augmented_train",
        num_augmentations=5
    )