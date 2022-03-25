from collections import namedtuple


class ImageExtension:
    JPEG = "jpeg"
    PNG = "png"


class Dataset:
    ROOT_FOLDER = "dataset"
    NORMAL = "normal"
    NOISY = "noisy"


class Subset:
    NORMAL = "normal"
    JPEG_COMPRESSION = "jpeg_compression"
    MILD_GAUSSIAN = "mild_gaussian"
    STRONG_GAUSSIAN = "strong_gaussian"
    MOTION_BLUR = "motion_blur"


class Descriptor:
    RAW = "raw"
    HOG = "hog"
    RESNET18 = "resnet18"


class Resnet18Layer:
    CONV1 = "conv1"
    LAYER1_0_CONV1 = "layer1[0].conv1"
    LAYER1_0_CONV2 = "layer1[0].conv2"
    LAYER1_1_CONV1 = "layer1[1].conv1"
    LAYER1_1_CONV2 = "layer1[1].conv2"
    LAYER2_0_CONV1 = "layer2[0].conv1"
    LAYER2_0_CONV2 = "layer2[0].conv2"
    LAYER2_1_CONV1 = "layer2[1].conv1"
    LAYER2_1_CONV2 = "layer2[1].conv2"
    LAYER3_0_CONV1 = "layer3[0].conv1"
    LAYER3_0_CONV2 = "layer3[0].conv2"
    LAYER3_1_CONV1 = "layer3[1].conv1"
    LAYER3_1_CONV2 = "layer3[1].conv2"
    LAYER4_0_CONV1 = "layer4[0].conv1"
    LAYER4_0_CONV2 = "layer4[0].conv2"
    LAYER4_1_CONV1 = "layer4[1].conv1"
    LAYER4_1_CONV2 = "layer4[1].conv2"


basic_params = namedtuple("basic_params", ["seed", "save_path"])
datasets_params = namedtuple("datasets_params", ["ground_truth_file_name", "dataset_path",
                                                 "image_extension", "image_width", "image_height",
                                                 "train_datasets", "test_dataset",
                                                 "test_subset", "test_percent"])
descriptor_params = namedtuple("descriptor_params", ["descriptor_algorithm", "params"])
hog_params = namedtuple("hog_params", ["features_per_cell", "block_size", "block_stride", "cell_size"])
resnet18_params = namedtuple("resnet18_params", ["layer"])
gp_params = namedtuple("gp_params", ["L0", "Sigma0"])
exp = namedtuple("exp", ["basic_params", "datasets_params", "descriptor_params", "gp_params"])


