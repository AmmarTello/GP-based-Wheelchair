from abc import ABCMeta, abstractmethod
import numpy as np


class FeatureExtractionInterface(metaclass=ABCMeta):
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
        file_name = self.get_file_name(dataset_name)

        self.__dict__.pop("extractor")

        np.savez(f"{file_name}.npz", descriptors=descriptors, images_names=images_names,
                 elapsed_time=elapsed_time, **self.__dict__)

    def load(self, dataset_name):
        file_name = self.get_file_name(dataset_name)

        return np.load(file_name, allow_pickle=True)

    def get_file_name(self, dataset_name):
        descriptor_details = self.get_descriptor_details()
        file_name = "_".join([dataset_name, self.DESCRIPTOR_ALGORITHM, descriptor_details, "descriptor"])

        return file_name


from .hog import HOG
from .resnet18 import ResNet18
