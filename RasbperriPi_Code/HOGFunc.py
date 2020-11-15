import cv2
import numpy as np
from Configuration import read_resize_image
import time


def hog_parameters():
    features_per_cell = 9
    block_size = (32, 32)
    block_stride = (32, 32)
    cell_size = (32, 32)

    return features_per_cell, block_size, block_stride, cell_size


def extended_hog_parameters(image_height, image_width):
    # HoG descriptor parameters
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = False
    nlevels = 64
    signedGradients = False
    winSize = (image_width, image_height)

    return derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients, winSize


def hog_func_one_image(image, image_height, image_width, hog, winSize):

    # resize_image = read_resize_image(image, image_width, image_height)
    start = time.time()
    descriptors = np.transpose(hog.compute(image, winSize))
    elapsed_time = time.time() - start

    return descriptors, elapsed_time

def hog_func(images_paths, image_height, image_width, dataset_name):

    # HoG descriptor parameters
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = False
    nlevels = 64
    signedGradients = False
    winSize = (image_width, image_height)

    features_per_cell, block_size, block_stride, cell_size = hog_parameters()

    block_size_in_cell = np.array([block_size])/np.array([cell_size])
    block_overlap = block_size_in_cell - np.array([block_stride])/np.array([cell_size])
    blocks_per_image = (np.divide(np.array([image_width, image_height]), np.array([cell_size]))
                        - block_size_in_cell)/(block_size_in_cell-block_overlap)+1
    number_of_cells = (np.prod(blocks_per_image) * np.prod(block_size_in_cell))
    hog_descriptor_length = int(number_of_cells * features_per_cell)

    hog = cv2.HOGDescriptor(winSize, block_size, block_stride, cell_size, features_per_cell, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)

    number_of_images = len(images_paths)
    descriptors = np.empty((number_of_images, hog_descriptor_length))
    images_labels = []
    elapsed_time = np.empty((number_of_images, 1))
    for ii, image_path in enumerate(images_paths):
        print(ii)
        image_name = image_path.split("\\")[-1].split(".")[0]
        images_labels.append(image_name)
        resize_image = read_resize_image(image_path, image_width, image_height)
        start = time.time()
        descriptors[ii, :] = np.transpose(hog.compute(resize_image, winSize))
        elapsed_time = time.time() - start

    descriptor_file = "_".join(["Descriptors/" + dataset_name, "Hog", str(features_per_cell), str(block_size[0]),
                                str(block_stride[0]), str(cell_size[0]),  "descriptor"])
    np.savez(descriptor_file, descriptors=descriptors, number_of_cells=number_of_cells,
             image_height=image_height, image_width=image_width, number_of_images=number_of_images,
             features_per_cell=features_per_cell, block_size=block_size, block_stride=block_stride,
             cell_size=cell_size, derivAperture=derivAperture, winSigma=winSigma, histogramNormType=histogramNormType,
             L2HysThreshold=L2HysThreshold, gammaCorrection=gammaCorrection, nlevels=nlevels,
             ignedGradients=signedGradients, winSize=winSize, images_labels=images_labels, elapsed_time=elapsed_time)


