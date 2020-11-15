"""
This script allows the user to apply noises randomly on a dataset by configuring
the path and the other noises related parameters in the "main" function.

"""

import cv2
import numpy as np
from glob import glob


def motion_blur(img, filter_size):

    kernel_motion_blur = np.zeros((filter_size, filter_size))
    kernel_motion_blur[int((filter_size-1)/2), :] = np.ones(filter_size)
    kernel_motion_blur = kernel_motion_blur / filter_size

    # applying the kernel to the input image
    img = cv2.filter2D(img, -1, kernel_motion_blur)

    return img


def jpeg_compression(img, image_name):
    image_name = image_name.replace("png", "jpeg")
    noised_image = img
    return noised_image, image_name


def main():
    np.random.seed(0)

    base_path = "..\\Dataset\\"

    # Noises Parameters
    blur_filter_size = 10
    strong_gaussian_filter = (5, 5)
    mild_gaussian_filter = (13, 13)

    first_image_label = 5612  # The label of the first image in the generated noised images

    noises_types = ["Motion blur", "Strong Gaussian", "Mild Gaussian", "JPEG compression", "Illumination"]

    images_paths = glob(base_path + "Normal\\*.png")

    for ii, image_path in enumerate(images_paths):
        print(ii)

        image_name = image_path.split("\\")[-1]
        img = cv2.imread(image_path)
        noise_type = noises_types[np.random.randint(0, len(noises_types))]

        if noise_type == "Motion blur":
            noised_image = motion_blur(img, blur_filter_size)
        elif noise_type == "Strong Gaussian":
            noised_image = cv2.GaussianBlur(img, strong_gaussian_filter, 0)
        elif noise_type == "Mild Gaussian":
            noised_image = cv2.GaussianBlur(img, mild_gaussian_filter, 0)
        elif noise_type == "JPEG compression":
            noised_image, image_name = jpeg_compression(img, image_name)
        else:
            noised_image = None
            print("No Noise Was Specified!!!")

        noised_image_path = base_path + "Noisy\\" + noise_type + "\\" + \
                            image_name.replace(image_name.split(".")[0], str(ii + first_image_label))
        cv2.imwrite(noised_image_path, noised_image)


if __name__ == "__main__":
    # execute only if run as a script
    main()
