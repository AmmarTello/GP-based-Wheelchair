from exp_preparation import *

# basic_params = namedtuple("basic_params", [])
# datasets_params = namedtuple("datasets_params", [])
# hog_params = namedtuple("hog_params", [])
# resnet18_params = namedtuple("resnet18_params", [])
# gp_params = namedtuple("gp_params", [])
# descriptor_params = namedtuple("descriptor_params", [])
# exp = namedtuple("exp", ["basic_params", "datasets_params", "descriptor_params", "gp_params"])


# Basic Params
basic_params.seed = 0
basic_params.save_path = "saved_files/results/"


# Dataset
datasets_params.ground_truth_velocities_file_name = "Ground Truth.csv"  # Velocities

datasets_params.dataset_path = Datasets.ROOT_FOLDER
datasets_params.image_extension = ImageExtension.PNG
datasets_params.image_width, datasets_params.image_height = (224, 224)

# Training Parameters
datasets_params.train_datasets = [Datasets.NORMAL, Datasets.NOISY]  # dataset Names ["normal", "noisy", "unreliable"]

# Testing Parameters
datasets_params.test_dataset = Datasets.NOISY  # None for test on all training datasets, or put one dataset index for test on one dataset

datasets_params.test_subset_flag = True   # If noisy
datasets_params.test_subset = Subsets.MILD_GAUSSIAN      # ["jpeg_compression", "mild_gaussian", "strong_gaussian", "motion_blur"]

datasets_params.test_percent = 0.1

# Descriptor
descriptor_params.descriptor_algorithm = FeatureExtractionAlgorithms.HOG  # ["Raw", "Hog", "ResNet18"]

# ResNet18 Parameters
resnet18_params.layer = Resnet18Layers.LAYER3_0_CONV2

# HOG Parameters
hog_params.features_per_cell = 9
hog_params.block_size = 32
hog_params.block_stride = 32
hog_params.cell_size = 32

# Train GP
gp_params.L0 = 0.5  # Initial length scale
gp_params.Sigma0 = 0.1  # Initial noise standard deviation


descriptor_params.params = eval(f"{descriptor_params.descriptor_algorithm}_params")

my_exp = exp(basic_params, datasets_params, descriptor_params, gp_params)



