# src/data/data_augmentation.py

import os
import cv2
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import albumentations as A
import logging
import uuid
from pathlib import Path

def create_dir(path):
    """Create a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def load_data(images_dir, annotations_dir):
    """Load image and annotation file paths."""
    image_extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.png"]  # Add more as needed
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(str(images_dir / ext)))
    image_paths = sorted(image_paths)
    logging.info(f"Found {len(image_paths)} images in {images_dir}")

    annotation_paths = []
    for img_path in image_paths:
        img_path = Path(img_path)
        ann_path = annotations_dir / (img_path.stem + '.txt')
        if ann_path.exists():
            annotation_paths.append(ann_path)
        else:
            logging.warning(f"Annotation file not found for {img_path.name}")
            annotation_paths.append(None)
    return list(zip(image_paths, annotation_paths))

def read_yolo_annotation(ann_path, img_width, img_height):
    """Read YOLO format annotations and convert them to format compatible with albumentations."""
    bboxes = []
    labels = []
    with open(ann_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                parts = line.strip().split()
                if len(parts) != 5:
                    raise ValueError(f"Incorrect number of fields: {line}")
                class_id, x_center, y_center, width, height = map(float, parts)
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height
                xmin = max(x_center - width / 2, 0)
                ymin = max(y_center - height / 2, 0)
                xmax = min(x_center + width / 2, img_width)
                ymax = min(y_center + height / 2, img_height)
                if xmin >= xmax or ymin >= ymax:
                    raise ValueError(f"Invalid bbox with zero area: {line}")
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(class_id))
            except Exception as e:
                logging.error(f"Error parsing line in {ann_path}: {e}")
                continue
    return bboxes, labels

def convert_bboxes_to_yolo(bboxes, img_width, img_height):
    """Convert bounding boxes to YOLO format."""
    yolo_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        yolo_bboxes.append([x_center, y_center, width, height])
    return yolo_bboxes

def save_yolo_annotation(ann_path, yolo_bboxes, labels):
    """Save annotations in YOLO format."""
    with open(ann_path, 'w') as f:
        for bbox, label in zip(yolo_bboxes, labels):
            bbox_line = f"{label} " + " ".join(f"{coord:.6f}" for coord in bbox)
            f.write(bbox_line + '\n')

def augment_data(data, save_img_dir, save_ann_dir, num_augments=5):
    """Perform data augmentation and save augmented images and annotations."""
    create_dir(save_img_dir)
    create_dir(save_ann_dir)

    # Define augmentation pipeline
    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MotionBlur(p=0.2),
        A.GaussNoise(p=0.2),
        A.HueSaturationValue(p=0.5),
        # Add more augmentations as needed
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    for idx, (img_path, ann_path) in tqdm(enumerate(data), total=len(data), desc="Augmenting Data"):
        base_name = Path(img_path).stem

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning(f"Unable to read image {img_path.name}")
            continue
        img_height, img_width = img.shape[:2]

        # Read annotation
        if ann_path is None:
            logging.info(f"No annotation for image {img_path.name}, skipping.")
            continue
        bboxes, labels = read_yolo_annotation(ann_path, img_width, img_height)

        if not bboxes:
            logging.info(f"No valid bounding boxes for image {img_path.name}, skipping augmentation.")
            continue

        # Prepare data for augmentation
        annotations = {
            'image': img,
            'bboxes': bboxes,
            'labels': labels
        }

        # Generate multiple augmentations per image
        for i in range(num_augments):
            augmented = augmentation_pipeline(**annotations)
            aug_img = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['labels']

            # Skip images with no bounding boxes after augmentation
            if not aug_bboxes:
                logging.info(f"No bounding boxes left after augmentation for {img_path.name}")
                continue

            aug_img_height, aug_img_width = aug_img.shape[:2]

            # Convert augmented bounding boxes to YOLO format
            yolo_bboxes = convert_bboxes_to_yolo(aug_bboxes, aug_img_width, aug_img_height)

            # Save augmented image with unique identifier
            unique_id = uuid.uuid4().hex
            aug_img_name = f"{base_name}_aug_{unique_id}.jpg"
            aug_img_save_path = save_img_dir / aug_img_name
            cv2.imwrite(str(aug_img_save_path), aug_img)

            # Save augmented annotation
            aug_ann_name = f"{base_name}_aug_{unique_id}.txt"
            aug_ann_save_path = save_ann_dir / aug_ann_name
            save_yolo_annotation(aug_ann_save_path, yolo_bboxes, aug_labels)

    logging.info("Data augmentation completed.")
