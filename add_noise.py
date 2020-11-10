import cv2
import numpy as np
from glob import glob


def motion_blur(img):
    size = 10
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    # applying the kernel to the input image
    img = cv2.filter2D(img, -1, kernel_motion_blur)

    return img


def gaussian_noise(img, level):
    if level == "Mild":
        gaussian_filter = (5, 5)

    elif level == "Strong":
        gaussian_filter = (13, 13)

    img = cv2.GaussianBlur(img, gaussian_filter, 0)

    return img


def jpeg_compression(img, image_name):
    image_name = image_name.replace("png", "jpeg")
    noised_image = img
    return noised_image, image_name


def illumination_change(img):
    factor = 0.2
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[..., 2] = hsv_img[..., 2] * factor

    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)


def main():
    np.random.seed(0)

    base_path = "H:\\Desktop\\CARA\\Dataset\\"

    noises_types = ["Motion blur", "Strong Gaussian", "Mild Gaussian", "JPEG compression", "Illumination"]

    images_paths = glob(base_path + "Normal\\*.png")

    for ii, image_path in enumerate(images_paths):
        print(ii)

        image_name = image_path.split("\\")[-1]
        img = cv2.imread(image_path)
        noise_type = noises_types[np.random.randint(0, len(noises_types))]
        noise_type = "Illumination"
        if ii > 672:
            break
        if noise_type == "Motion blur":
            noised_image = motion_blur(img)
        elif noise_type == "Strong Gaussian":
            noised_image = gaussian_noise(img, "Strong")
        elif noise_type == "Mild Gaussian":
            noised_image = gaussian_noise(img, "Mild")
        elif noise_type == "JPEG compression":
            noised_image, image_name = jpeg_compression(img, image_name)
        elif noise_type == "Illumination":
            noised_image = illumination_change(img)
        else:
            noised_image = None
            print("No Noise Was Specified!!!")

        noised_image_path = base_path + "Noisy\\" + noise_type + "\\" + \
                            image_name.replace(image_name.split(".")[0], str(ii + 5612))
        cv2.imwrite(noised_image_path, noised_image)


if __name__ == "__main__":
    # execute only if run as a script
    main()
