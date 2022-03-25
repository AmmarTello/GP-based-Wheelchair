"""
This module is responsible of extracting the descriptors for a dataset.
Just run this module (after configuring the experiment) in order to extract the descriptors.
The results will be saved under "saved_files/descriptors"
"""

import os
import time
import torchvision.transforms as transforms
import cv2

from configuration import my_exp
from descriptors import HOG, ResNet18
from data_preparation.transforms import ColorConversion, ToOpenCV, Resize, Transform
from data_preparation.dataset import ImageDataset


def main():
    datasets_paths = []
    for dataset_name in my_exp.datasets_params.train_datasets:
        datasets_paths.append(os.path.join(my_exp.datasets_params.dataset_path, dataset_name))

    if my_exp.descriptor_params.descriptor_algorithm == "hog":
        list_of_transforms = [ToOpenCV(),
                              ColorConversion(cv2.COLOR_BGR2GRAY),
                              Resize(my_exp.datasets_params.image_height, my_exp.datasets_params.image_width)]
        descriptor_algorithm = HOG(my_exp.datasets_params.image_height, my_exp.datasets_params.image_width)

    elif my_exp.descriptor_params.descriptor_algorithm == "resnet18":
        list_of_transforms = [transforms.Resize((my_exp.datasets_params.image_height,
                                                 my_exp.datasets_params.image_width)),
                              transforms.ToTensor()]
        descriptor_algorithm = ResNet18(my_exp.descriptor_params.params.layer, pretrained=True)

    else:
        raise Exception("This descriptor is not supported!")

    image_dataset = ImageDataset(datasets_paths,
                                 my_exp.datasets_params.ground_truth_file_name,
                                 transform=Transform(list_of_transforms))

    start = time.time()

    descriptors = []
    images_names = []
    for index, (image, velocity, image_name) in enumerate(image_dataset):
        descriptors.append(descriptor_algorithm(image))
        images_names.append(image_name)
        print(index)

    elapsed_time = time.time() - start

    descriptor_algorithm.save(my_exp.datasets_params.train_datasets[0], descriptors, images_names, elapsed_time)


if __name__ == "__main__":
    main()
