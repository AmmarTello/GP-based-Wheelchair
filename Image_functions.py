from glob import glob
import cv2
import os
import numpy as np


def read_resize_image(image_path, image_width=None, image_height=None, color_flag=0):
    image = cv2.imread(image_path, color_flag)
    if image_width is not None:
        image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_NEAREST)

    return image


def collect_images_paths(path, image_extension, number_of_images=None):
    subfolders_paths = [f.name for f in os.scandir(path) if f.is_dir()]
    images_paths = np.array([])
    if len(subfolders_paths) == 0:
        image_name = path + "*." + image_extension
        images_paths = np.array(glob(image_name))
    else:
        for folder_name in subfolders_paths:
            image_name = path + folder_name + "\\*"
            images_paths = np.concatenate([images_paths, np.array(glob(image_name))])
    if number_of_images:
        images_paths = images_paths[:number_of_images]
    number_of_images = len(images_paths)

    images_labels = np.array([int(image_path.split("\\")[-1].split(".")[0]) for image_path in images_paths])
    sorted_labels_indices = np.argsort(images_labels)

    return number_of_images, images_paths[sorted_labels_indices], images_labels[sorted_labels_indices]
