import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from albumentations import HorizontalFlip, Rotate


# Create directory
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# load data
def load_data(image_path, annotation_path):
    images = sorted(glob(os.path.join(image_path, "*.JPG")))  # Adjust extension if needed
    annotations = sorted(glob(os.path.join(annotation_path, "*.xml")))
    return images, annotations


# Save augmented data
def save_augmented_data(aug_images, aug_annotations, save_img_dir, save_ann_dir, base_name, index):
    for i, (img, ann) in enumerate(zip(aug_images, aug_annotations)):
        img_name = f"{base_name}_{index + i}.png"
        annotation_name = f"{base_name}_{index + i}.xml"
        
        cv2.imwrite(os.path.join(save_img_dir, img_name), img)
        # Assuming annotations are copied; modify if augmentation changes them
        with open(os.path.join(save_ann_dir, annotation_name), 'w') as f:
            f.write(ann)


# Data augmentation
def augment_data(images, annotations, save_img_dir, save_ann_dir, augment=True):
    size = (512, 512)
    create_dir(save_img_dir)
    create_dir(save_ann_dir)

    for ind, (img_path, ann_path) in tqdm(enumerate(zip(images, annotations)), total=len(images)):
        base_name = os.path.basename(img_path).split(".")[0]

        # Read image and annotation
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        with open(ann_path, 'r') as f:
            annotation = f.read()

        if augment:
            # Apply augmentations
            aug1 = HorizontalFlip(p=1.0)
            aug_img1 = aug1(image=img)['image']

            aug2 = Rotate(limit=45, p=1.0)
            aug_img2 = aug2(image=img)['image']

            aug_images = [img, aug_img1, aug_img2]
            aug_annotations = [annotation] * len(aug_images)  # Assume annotations remain the same for these augmentations
        else:
            aug_images = [img]
            aug_annotations = [annotation]

        # Resize and save augmented images and annotations
        resized_images = [cv2.resize(aug_img, size) for aug_img in aug_images]
        save_augmented_data(resized_images, aug_annotations, save_img_dir, save_ann_dir, base_name, ind * len(aug_images))


if __name__ == "__main__":
    """ Load the data"""
    repo_root = os.path.dirname(os.path.abspath(__file__))  # Get repo root dynamically
    images_path = os.path.join(repo_root, "../data/full_dataset/images")
    annotations_path = os.path.join(repo_root, "../data/full_dataset/annotations")
    save_img_dir = os.path.join(repo_root, "../data/augmented_dataset/images")
    save_ann_dir = os.path.join(repo_root, "../data/augmented_dataset/annotations")

    images, annotations = load_data(images_path, annotations_path)

    """ Data augmentation"""
    augment_data(images, annotations, save_img_dir, save_ann_dir, augment=True)
