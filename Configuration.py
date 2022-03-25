from utilities.constants import *


# Basic Params
basic_params.seed = 0
basic_params.save_path = "saved_files/results/"


# Dataset
datasets_params.ground_truth_file_name = "Ground Truth.csv"  # Velocities

datasets_params.dataset_path = Dataset.ROOT_FOLDER
datasets_params.image_extension = ImageExtension.PNG
datasets_params.image_width, datasets_params.image_height = (224, 224)

# Training Parameters
datasets_params.train_datasets = [Dataset.NORMAL]  # dataset Names ["NORMAL", "NOISY"]

# Testing Parameters
datasets_params.test_dataset = None  # None for test on a subset of training datasets, or Dataset.NORMAL, Dataset.NOISY
datasets_params.test_subset = Subset.MILD_GAUSSIAN  # ["JPEG_COMPRESSION", "MILD_GAUSSIAN", "STRONG_GAUSSIAN", "MOTION_BLUR"]

datasets_params.test_percent = 0.1

# Descriptor
descriptor_params.descriptor_algorithm = Descriptor.HOG  # ["HOG", "RESNET18"]

# ResNet18 Parameters
resnet18_params.layer = Resnet18Layer.LAYER3_0_CONV2

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



