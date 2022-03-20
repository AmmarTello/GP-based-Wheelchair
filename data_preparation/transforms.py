from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
import torchvision.transforms as transforms


class TransformerInterface(metaclass=ABCMeta):

    @abstractmethod
    def transform(self, image):
        pass

    def __call__(self, image):
        return self.transform(image)


class Resize(TransformerInterface):

    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

    def transform(self, image):
        return cv2.resize(image, dsize=(self.image_width, self.image_height),
                          interpolation=cv2.INTER_NEAREST)


class ColorConversion(TransformerInterface):
    def __init__(self, color_flag):
        self.color_flag = color_flag

    def transform(self, image):
        return cv2.cvtColor(image, self.color_flag)


class ToOpenCV(TransformerInterface):

    def transform(self, image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


class Transform(TransformerInterface):
    def __init__(self, list_of_transforms):
        self.list_of_transforms = list_of_transforms

    def transform(self, image):
        return transforms.Compose(self.list_of_transforms)(image)

