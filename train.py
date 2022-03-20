import numpy as np

from gp.gp import GP
from feature_extraction import HOG, ResNet18
from dataset_preparation import DatasetPreparation
from configuration import my_exp
from sklearn.metrics import r2_score
from data_preparation.transforms import *
from utilities.save_files import FilesPaths, LoadResults
import os


np.random.seed(my_exp.basic_params.seed)

datasets_paths = []
for dataset_name in my_exp.datasets_params.train_datasets:
    datasets_paths.append(os.path.join(my_exp.datasets_params.dataset_path, dataset_name))


# Dataset preparation
dataset_prep = DatasetPreparation(my_exp.datasets_params.dataset_path, datasets_paths,
                                  my_exp.datasets_params.test_dataset, my_exp.datasets_params.test_subset,
                                  my_exp.datasets_params.ground_truth_velocities_file_name,
                                  my_exp.datasets_params.test_percent, my_exp.basic_params.seed)

train_dataset_dict, test_dataset_dict = dataset_prep()

# Load descriptors
# HOG
if my_exp.descriptor_params.descriptor_algorithm == "hog":
    descriptor_algorithm = HOG(my_exp.datasets_params.image_height, my_exp.datasets_params.image_width)

# ResNet18
elif my_exp.descriptor_params.descriptor_algorithm == "resnet18":
    descriptor_algorithm = ResNet18(my_exp.descriptor_params.params.layer, pretrained=True)

else:
    raise Exception("This descriptor is not supported!")

training_descriptors = LoadResults.load_descriptors(my_exp.datasets_params.train_datasets,
                                                    descriptor_algorithm,
                                                    train_dataset_dict["training_labels"])

train_gp_dict = GP.train(my_exp.gp_params.L0, my_exp.gp_params.Sigma0, training_descriptors,
                         train_dataset_dict["training_velocities"])

print('Hyperparameters:', train_gp_dict["L"], train_gp_dict["Sigma"])
print('Elapsed Time:', train_gp_dict["elapsed_time"])

# Test GP
testing_descriptors = LoadResults.load_descriptors(my_exp.datasets_params.train_datasets,
                                                   descriptor_algorithm,
                                                   test_dataset_dict["testing_labels"])

test_gp_dict = GP.test(training_descriptors, testing_descriptors, train_gp_dict,
                       train_dataset_dict["training_velocities"])


# Evaluation
metrics = {}
metrics["mse"] = np.mean((test_gp_dict["Y_StarMean"] - test_dataset_dict["testing_velocities"]) ** 2)
print(metrics["mse"])

metrics["r_score"] = r2_score(test_dataset_dict["testing_velocities"], test_gp_dict["Y_StarMean"])
print(metrics["r_score"])

descriptor_details = descriptor_algorithm.get_descriptor_details()

gp_file_path = FilesPaths.get_gp_file_path(my_exp.datasets_params.train_datasets,
                                           my_exp.datasets_params.test_dataset, descriptor_details)

GP.save(gp_file_path, my_exp, train_gp_dict, test_gp_dict, train_dataset_dict, test_dataset_dict, metrics)
