import cv2
import numpy as np

from . import DescriptorInterface
from configuration import Descriptor


class HOG(DescriptorInterface):
    """
    The HOG descriptor
    """
    DESCRIPTOR_ALGORITHM = Descriptor.HOG

    def __init__(self, image_width, image_height, features_per_cell=9, block_size=(32, 32),
                 block_stride=(32, 32), cell_size=(32, 32), deriv_aperture=1, win_sigma=-1.,
                 histogram_norm_type=0, l2_hys_threshold=0.2, gamma_correction=False,
                 n_levels=64, signed_gradients=False):
        super().__init__()

        # Basic parameters
        self.features_per_cell = features_per_cell
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size

        # Advanced parameters
        self.deriv_aperture = deriv_aperture
        self.win_sigma = win_sigma
        self.histogram_norm_type = histogram_norm_type
        self.l2_hys_threshold = l2_hys_threshold
        self.gamma_correction = gamma_correction
        self.n_levels = n_levels
        self.signed_gradients = signed_gradients
        self.image_width = image_width
        self.image_height = image_height
        self.win_Size = (image_width, image_height)

        self.extractor = cv2.HOGDescriptor(self.win_Size, self.block_size, self.block_stride, self.cell_size,
                                           self.features_per_cell, self.deriv_aperture, self.win_sigma,
                                           self.histogram_norm_type, self.l2_hys_threshold, self.gamma_correction,
                                           self.n_levels, self.signed_gradients)

    def extract(self, image):
        return np.transpose(self.extractor.compute(image, self.win_Size))

    def calculate_hog_descriptor_length(self):
        block_size_in_cell = np.array([self.block_size]) / np.array([self.cell_size])
        block_overlap = block_size_in_cell - np.array([self.block_stride]) / np.array([self.cell_size])
        blocks_per_image = (np.divide(np.array([self.image_width, self.image_height]), np.array([self.cell_size]))
                            - block_size_in_cell) / (block_size_in_cell - block_overlap) + 1
        number_of_cells = (np.prod(blocks_per_image) * np.prod(block_size_in_cell))
        hog_descriptor_length = int(number_of_cells * self.features_per_cell)

        return hog_descriptor_length

    def get_descriptor_details(self):
        descriptor_details = [str(self.features_per_cell), str(self.block_size[0]),
                              str(self.block_stride[0]), str(self.cell_size[0])]

        return "_".join(descriptor_details)
