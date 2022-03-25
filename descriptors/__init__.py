import os
from abc import ABCMeta, abstractmethod
import numpy as np


class DescriptorInterface(metaclass=ABCMeta):
    """
    Interface for descriptors algorithms. All descriptors classes should follow this interface.
    """

    DESCRIPTOR_FOLDER = "saved_files/descriptors"
    DESCRIPTOR_ALGORITHM = None

    def __init__(self):
        self.extractor = None

    def __call__(self, image):
        return self.extract(image)

    @abstractmethod
    def extract(self, image):
        pass

    @abstractmethod
    def get_descriptor_details(self):
        pass

    def save(self, dataset_name, descriptors, images_names, elapsed_time):
        file_name = self.get_file_path(dataset_name)

        if "extractor" in self.__dict__:
            self.__dict__.pop("extractor")

        for param in list(self.__dict__):
            if param.startswith("_"):
                self.__dict__.pop(param)

        np.savez(f"{file_name}.npz", descriptors=descriptors, images_names=images_names,
                 elapsed_time=elapsed_time, **self.__dict__)

    def load(self, dataset_name):
        file_name = self.get_file_path(dataset_name)

        return np.load(f"{file_name}.npz", allow_pickle=True)

    def get_file_path(self, dataset_name):
        descriptor_details = self.get_descriptor_details()
        file_name = "_".join([dataset_name, self.DESCRIPTOR_ALGORITHM, descriptor_details, "descriptor"])

        file_path = os.path.join(self.DESCRIPTOR_FOLDER, file_name)

        return file_path


from .hog import HOG
from .resnet18 import ResNet18
