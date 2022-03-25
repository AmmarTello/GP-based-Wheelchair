import os
import numpy as np
from sklearn.metrics import r2_score

from configuration import my_exp
from data_preparation import DatasetSplitting
from descriptors import HOG, ResNet18
from gp import GP
from utilities import LoadResults


# Seed
np.random.seed(my_exp.basic_params.seed)

# Dataset preparation
datasets_paths = []
for dataset_name in my_exp.datasets_params.train_datasets:
    datasets_paths.append(os.path.join(my_exp.datasets_params.dataset_path, dataset_name))


dataset_prep = DatasetSplitting(my_exp.datasets_params.dataset_path, datasets_paths,
                                my_exp.datasets_params.test_dataset, my_exp.datasets_params.test_subset,
                                my_exp.datasets_params.ground_truth_file_name,
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

training_descriptors, training_velocities = LoadResults.load_descriptors(my_exp.datasets_params.train_datasets,
                                                                         descriptor_algorithm,
                                                                         train_dataset_dict["train_images_names"],
                                                                         train_dataset_dict["train_velocities"])

# Train GP
train_gp_dict = GP.train(my_exp.gp_params.L0, my_exp.gp_params.Sigma0, training_descriptors, training_velocities)

print('Hyperparameters:', train_gp_dict["L"], train_gp_dict["Sigma"])
print('Elapsed Time:', train_gp_dict["elapsed_time"])

# Test GP
testing_descriptors, testing_velocities = LoadResults.load_descriptors(my_exp.datasets_params.train_datasets,
                                                                       descriptor_algorithm,
                                                                       test_dataset_dict["test_images_names"],
                                                                       test_dataset_dict["test_velocities"])

test_gp_dict = GP.test(training_descriptors, testing_descriptors, train_gp_dict,
                       train_dataset_dict["train_velocities"])


# Evaluation
metrics = {}

metrics["mse"] = np.mean((test_gp_dict["Y_StarMean"] - testing_velocities) ** 2)
print("mse:", metrics["mse"])

metrics["r_score"] = r2_score(testing_velocities, test_gp_dict["Y_StarMean"])
print("r_score:", metrics["r_score"])


# Save Results
descriptor_details = descriptor_algorithm.get_descriptor_details()

GP.save(my_exp.datasets_params.train_datasets, my_exp.datasets_params.test_dataset,
        descriptor_details, my_exp, train_gp_dict, test_gp_dict, train_dataset_dict,
        test_dataset_dict, metrics)
