import numpy as np
import LIRU_GP2 as GP
from Configuration import load_desc, exp_parameters, files_names
from Image_functions import collect_images_paths, read_resize_image
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import r2_score
import LIRU_SparseGP as sGP


exp = exp_parameters()
np.random.seed(exp["seed"])

dataset_path = "..\\Dataset\\"
image_extension = "png"

save_path = "Results\\"
saved_file_name, desc_file, test_dataset_path = files_names(exp, dataset_path, save_path)

ground_truth_velocities_file_name = "Ground Truth.csv"  # Velocities
with open(ground_truth_velocities_file_name) as csv_file:
    velocities_file = csv.reader(csv_file, delimiter=',')
    velocities_file_content = np.array([[int(row[0].split(".")[0]), float(row[1])] for row in velocities_file])

sorter = np.argsort(velocities_file_content[:, 0])

images_paths, velocities, images_labels = [], [], []
for ii in exp["train_datasets"]:
    number_of_images, current_images_paths, current_images_labels = collect_images_paths(dataset_path + ii + "\\", image_extension)
    images_paths = np.concatenate([images_paths, current_images_paths])
    labels_indices = sorter[np.searchsorted(velocities_file_content[:, 0], current_images_labels, sorter=sorter)]
    velocities = np.concatenate([velocities, velocities_file_content[labels_indices, 1]])
    images_labels = np.concatenate([images_labels, current_images_labels])

if exp["test_subset"] is None:
    training_images_paths, testing_images_paths, \
        training_labels, testing_labels, \
        training_velocities, testing_velocities = \
        train_test_split(images_paths, images_labels, velocities, test_size=exp["test_percent"], random_state=exp["seed"])
else:
    if exp["test_subset"] == "JPEG_Compression":
        image_extension = "jpeg"

    _, sub_images_paths, sub_images_labels = collect_images_paths(test_dataset_path, image_extension)
    labels_indices = sorter[np.searchsorted(velocities_file_content[:, 0],
                                            sub_images_labels, sorter=sorter)]
    number_of_testing_images = round(exp["test_percent"] * len(images_labels))
    test_percent = number_of_testing_images / len(labels_indices)
    _, testing_images_paths, _, testing_labels, _, testing_velocities = \
        train_test_split(sub_images_paths, sub_images_labels, velocities_file_content[labels_indices, 1],
                         test_size=exp["test_percent"], random_state=exp["seed"])
    test_indices = np.invert(np.isin(images_labels, testing_labels))
    training_images_paths = images_paths[test_indices]
    training_labels = images_labels[test_indices]
    training_velocities = velocities[test_indices]

velocities_mean = np.mean(training_velocities)
velocities_std = np.std(training_velocities)
training_velocities = (training_velocities - velocities_mean) / velocities_std
testing_velocities = (testing_velocities - velocities_mean) / velocities_std

number_of_training_images = len(training_images_paths)
number_of_testing_images = len(testing_images_paths)

# Load descriptors
ground_truth_labels = []
descriptors = []
if exp["features_extraction_method"] in ("Hog", "ResNet18"):
    for dataset, jj in enumerate(exp["train_datasets"]):
        descriptors, ground_truth_labels = load_desc(desc_file)
        if exp["train_two_dataset_flag"] and exp["second_train_dataset"] == "Noisy":
            current_descriptors, current_ground_truth_labels = load_desc(desc_file[jj])
            descriptors = np.concatenate([descriptors, current_descriptors], 0)
            ground_truth_labels = np.concatenate([ground_truth_labels, current_ground_truth_labels])

        training_images = np.array([descriptors[ground_truth_labels == str(label), :] for label in training_labels]).squeeze()
    del descriptors

elif exp["features_extraction_method"] == "Raw":
    training_images = np.empty((number_of_training_images, exp["image_width"] * exp["image_height"]))
    for ii in range(number_of_training_images):
        print("Training Image:", ii)
        training_images[ii, :] = read_resize_image(training_images_paths[ii],
                                                   exp["image_width"], exp["image_height"]).flatten()


# Train GP
L0 = exp["L0"]        # Initial length scale
Sigma0 = exp["Sigma0"]   # Initial noise standard deviation

if exp["GP_type"] == "GP":
    L, Sigma, K, C, InvC, train_elapsed_time = \
        GP.Train(L0, Sigma0, training_images, training_velocities, number_of_training_images)  # Train GP
elif exp["GP_type"] == "GP_Sparse":
    M = exp["M"]      # No. sparse points
    NoCandidates = exp["NoCandidates"]   # No. of candidate sets of sparse points analysed
    L, Sigma, K, C, InvC, Xs, Ys, LB_best, train_elapsed_time = \
        sGP.Train(L0, Sigma0, training_images, training_velocities, number_of_training_images, M, NoCandidates)
    training_images = Xs
    training_velocities = Ys
    number_of_training_images = M

print('Hyperparameters:', L, Sigma)      # Print hyperparameters
print('Elapsed Time:', train_elapsed_time)     # Print time taken to train GP


##### Test Section
ground_truth_labels = []
descriptors = []
if exp["features_extraction_method"] in ("Hog", "ResNet18"):
    for dataset, jj in enumerate(exp["train_datasets"]):
        descriptors, ground_truth_labels = load_desc(desc_file)
        if exp["train_two_dataset_flag"] and exp["second_train_dataset"] == "Noisy":
            current_descriptors, current_ground_truth_labels = load_desc(desc_file[jj])
            descriptors = np.concatenate([descriptors, current_descriptors], 0)
            ground_truth_labels = np.concatenate([ground_truth_labels, current_ground_truth_labels])

    test_features = np.array([descriptors[ground_truth_labels == str(label), :] for label in testing_labels]).squeeze()
    del descriptors

# Make some predictions
Y_StarMean = np.zeros([number_of_testing_images])         # mean of GP predictions
Y_StarStd = np.zeros([number_of_testing_images])          # std of GP predictions
test_elapsed_time = np.zeros([number_of_testing_images])
for ii in range(number_of_testing_images):
    print("Testing Image:", ii)
    if exp["features_extraction_method"] in ("Hog", "ResNet18"):
        test_descriptor = test_features[ii, :]
    else:
        test_descriptor = read_resize_image(testing_images_paths[ii], exp["image_width"], exp["image_height"]).flatten()
    Y_StarMean[ii], Y_StarStd[ii], test_elapsed_time[ii] = GP.Predict(training_images, test_descriptor, L, Sigma,
                                                                      training_velocities, K, C, InvC, number_of_training_images)


mean_squared_error = np.mean((Y_StarMean - testing_velocities) ** 2)
print(mean_squared_error)

r_score = r2_score(testing_velocities, Y_StarMean)
print(r_score)


np.savez(saved_file_name, exp=exp, L=L, Sigma=Sigma, K=K,
         C=C, InvC=InvC, train_elapsed_time=train_elapsed_time, Y_StarMean=Y_StarMean,
         Y_StarStd=Y_StarStd, test_elapsed_time=test_elapsed_time, squared_mean_error=mean_squared_error,
         training_labels=training_labels, testing_labels=testing_labels, r_score=r_score,
         training_velocities=training_velocities, testing_velocities=testing_velocities,
         velocities_mean=velocities_mean, velocities_std=velocities_std)
