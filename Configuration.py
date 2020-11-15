import numpy as np
from CNN import CNNParameters
from HOGFunc import hog_parameters


def exp_parameters():
    exp = dict()

    exp["seed"] = 0

    features_extraction_index = 1  # ["Raw", "Hog", "ResNet18"]

    # Training Parameters
    train_dataset_indices = [0, 1, 2]  # Dataset Names ["Normal", "Noisy", "Unreliable"]

    # Testing Parameters
    test_dataset_indices = [0, 1, 2]  # Dataset Names ["Normal", "Noisy", "Unreliable"]

    exp["test_subset_flag"] = 0   # If Noisy
    test_subset_index = 0         # ["JPEG_Compression", "Mild_Gaussian", "Strong_Gaussian", "Motion_Blur"]

    exp["test_percent"] = 0.1

    # CNN Parameters
    exp["layer_Index"] = 16

    exp["features_per_cell"] = 9
    exp["block_size"] = (32, 32)
    exp["block_stride"] = (32, 32)
    exp["cell_size"] = (32, 32)

    # GP
    GP_type_index = 0  # ["GP", "GP_Sparse"]

    # Train GP
    exp["L0"] = 0  # Initial length scale
    exp["Sigma0"] = 0.1  # Initial noise standard deviation

    # Train sparse GP
    exp["M"] = 700  # No. sparse points
    exp["NoCandidates"] = 1000  # No. of candidate sets of sparse points analysed

    # Image Shape
    exp["image_width"], exp["image_height"] = (224, 224)

    # Assignments according to the above configuration
    all_features_methods = ["Raw", "Hog", "ResNet18"]
    all_datasets = np.array(["Normal", "Noisy", "Unreliable"])
    test_subsets = np.array(["Normal", "JPEG_Compression", "Mild_Gaussian",
                             "Strong_Gaussian", "Motion_Blur"])
    GP_types = ["GP", "GP_Sparse"]

    exp["GP_type"] = GP_types[GP_type_index]

    exp["features_extraction_method"] = all_features_methods[features_extraction_index]

    exp["train_datasets"] = all_datasets[train_dataset_indices]

    exp["test_dataset"] = all_datasets[test_dataset_indices]
    exp["test_subset"] = test_subsets[test_subset_index]

    return exp


def files_names(exp, dataset_path, save_path):
    train_datasets = "_".join(exp["train_datasets"])

    if exp["test_subsets"] is not None:
        exp["test_datasets"][exp == "Noisy"] = exp["test_subset"]
        test_dataset_path = dataset_path + "Noisy\\" + exp["test_subset"] + "\\"
    else:
        test_dataset_path = dataset_path + exp["test_dataset"] + "\\"
    test_dataset = "_".join(exp["test_dataset"])

    if exp["features_extraction_method"] == "Hog":
        features_per_cell, block_size, block_stride, cell_size = hog_parameters()
        features_details = "_".join(["Hog", str(features_per_cell), str(block_size[0]),
                                    str(block_stride[0]), str(cell_size[0])])

    elif exp["features_extraction_method"] == "ResNet18":
        layer_name = CNNParameters(exp["features_extraction_method"], exp["layer_Index"])[0]
        features_details = "_".join([exp["features_extraction_method"], layer_name.replace(".", "_")])

    else:
        features_details = exp["features_extraction_method"]

    desc_file = []

    for dataset in exp["train_datasets"]:
        desc_file.append("Descriptors/" + dataset + "_" + features_details + "_descriptor.npz")

    saved_file_name = save_path + "\\" + exp["GP_type"] + "_train_" + train_datasets + "_test_" + test_dataset \
                      + "_" + features_details + ".npz"

    return saved_file_name, desc_file, test_dataset_path


def load_desc(descriptor_file):
    data = np.load(descriptor_file, allow_pickle=True)
    descriptors = data["descriptors"]
    ground_truth_labels = data["images_labels"]

    return descriptors, ground_truth_labels
