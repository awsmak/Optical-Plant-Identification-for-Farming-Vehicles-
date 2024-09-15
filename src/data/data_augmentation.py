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
def load_data(path):
    train = sorted(glob(os.path.join(path, "*.JPG")))
    print(f"train: {len(train)}")
    return train


# data augmentation
def augment_data(images, save_path, augment=True):
    size = (512, 512)

    for ind, x in tqdm(enumerate(images), total=len(images)):
        # Extract the name
        name = x.split("/")[-1].split(".")[0]

        # Read image and mask
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        print(f"Loaded image {x.shape}: {x.dtype}")

        if augment:

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x)
            x1 = augmented['image']

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x)
            x2 = augmented['image']

            X = [x, x1, x2]


        else:

            X = [x]

        # Resizing images and masks
        index = 0
        for i in X:
            i = cv2.resize(i, size)

            tmp_image_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, tmp_image_name)

            cv2.imwrite(image_path, i)
            print(f"Saved image {i.shape}: {i.dtype} to {image_path}")

            index += 1


if __name__ == "__main__":
    """ Seeding """

    np.random.seed(42)

    """ Load the data"""
    data_path = "/home/akash/PycharmProjects/Plant_Detection/dataset_raw/test"
    val = load_data(data_path)

    print(f"train: {len(val)}")

    """Create directories to save the augmented data"""

    create_dir("augmented_data/train/")

    # Data augmentation
    augment_data(val, "augmented_data/test/", augment=False)