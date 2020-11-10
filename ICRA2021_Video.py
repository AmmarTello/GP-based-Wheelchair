import time
import cv2
from HOGFunc import hog_parameters, extended_hog_parameters
from GP_New import Predict
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import csv
from Image_functions import collect_images_paths
from sklearn.metrics import r2_score


exp = 3
image_height = 480
image_width = 480
fileName = "Results\\GP_train_Normal_Noisy_Unreliable_test_Normal_Noisy_Unreliable_Hog_9_32_32_32_old.npz"
normal_descriptor_name = "Descriptors\\Normal_Hog_9_32_32_32_descriptor.npz"
noisy_descriptor_name = "Descriptors\\Noisy_Hog_9_32_32_32_descriptor.npz"
unreliable_descriptor_name = "Descriptors\\Unreliable_Hog_9_32_32_32_descriptor.npz"

with np.load("HKU_angle_3.npz") as results:
    w = results["Y_StarMean"]
    end = results["test_elapsed_time"]
    training_velocities = results["training_velocities"]
    velocities_mean = results["velocities_mean"]
    velocities_std = results["velocities_std"]

end = np.cumsum(end)
with np.load(fileName) as results:
    L = results["L"]
    Sigma = results["Sigma"]
    K = results["K"]
    C = results["C"]
    InvC = results["InvC"]
    # training_velocities = results["training_velocities"]
    training_labels = results["training_labels"]

number_of_training_images = len(training_velocities)

with np.load(normal_descriptor_name) as normal_descriptor_file:
    normal_descriptor = normal_descriptor_file["descriptors"]
    normal_ground_truth_labels = normal_descriptor_file["images_labels"]
with np.load(noisy_descriptor_name) as noisy_descriptor_file:
    noisy_descriptor = noisy_descriptor_file["descriptors"]
    noisy_ground_truth_labels = noisy_descriptor_file["images_labels"]
with np.load(unreliable_descriptor_name) as unreliable_descriptor_file:
    unreliable_descriptor = unreliable_descriptor_file["descriptors"]
    unreliable_ground_truth_labels = unreliable_descriptor_file["images_labels"]

# velocities_mean = np.mean(training_velocities)
# velocities_std = np.std(training_velocities)
training_velocities = (training_velocities - velocities_mean) / velocities_std

ground_truth_labels = np.concatenate([normal_ground_truth_labels, noisy_ground_truth_labels, unreliable_ground_truth_labels])

training_descriptors = np.concatenate([normal_descriptor, noisy_descriptor, unreliable_descriptor], 0)
training_descriptors = np.array([training_descriptors[ground_truth_labels == str(label), :] for label in training_labels]).squeeze().T

ground_truth_velocities_file_name = "Ground Truth.csv"  # Velocities
with open(ground_truth_velocities_file_name) as csv_file:
    velocities_file = csv.reader(csv_file, delimiter=',')
    velocities_file_content = np.array([[int(row[0].split(".")[0]), float(row[1])] for row in velocities_file])

dataset_path = "..\\Dataset\\Campus\\"
number_of_images, images_paths, images_labels = collect_images_paths(dataset_path + str(exp) + "\\", "png")

sorter = np.argsort(velocities_file_content[:, 0])
labels_indices = sorter[np.searchsorted(velocities_file_content[:, 0], images_labels, sorter=sorter)]
testing_velocities = velocities_file_content[labels_indices, 1]

features_per_cell, block_size, block_stride, cell_size = hog_parameters()
derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, \
nlevels, signedGradients, winSize = extended_hog_parameters(image_height, image_width)

hog = cv2.HOGDescriptor(winSize, block_size, block_stride, cell_size, features_per_cell, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)

InvCxY = np.dot(InvC, training_velocities)

# end = []
counter = 0

# w = []
fig, ax = plt.subplots()
plt.ion()
plt.show()
plt.ylim(-4, 4)
plt.xlabel("Time (s)", fontsize=24)
plt.ylabel("$\omega$ (rad/s)", fontsize=24)
plt.xticks()
# plt.legend("w")
a1 = deque([0]*50)
b1 = deque([0]*50)
line, = plt.plot(a1)
plt.show()
for images_path in images_paths:

    image = cv2.imread(images_path)
    start_cycle = time.time()
    cv2.imshow("Frame", image)
    image = cv2.resize(image, (image_width, image_height))

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.medianBlur(image, 3)  # Removes salt and pepper noise by convolving the image with a (3,3) square kernel
    # image = cv2.GaussianBlur(image, (3, 3), 0)

    descriptor = hog.compute(image, winSize)

    # Y_StarMean = Predict(training_descriptors, descriptor, L, InvCxY)
    # w.append(Y_StarMean)
    # Y_StarMean = Y_StarMean * velocities_std + velocities_mean

    limit_x_1 = 5
    limit_x_2 = 0.5
    # end.append(time.time() - start_cycle)
    print("Time = ", end[counter])
    sumTime = end[counter]
    plt.xlim(sumTime-limit_x_1, sumTime + limit_x_2)
    a1.appendleft(w[counter])
    datatoplot = a1.pop()
    b1.appendleft(sumTime)
    datatoplot2 = b1.pop()
    line.set_ydata(a1)
    line.set_xdata(b1)
    minor_ticks = np.arange(sumTime-limit_x_1, sumTime + limit_x_2, limit_x_2)
    major_ticks = np.arange(sumTime-limit_x_1, sumTime + limit_x_2, limit_x_2)


    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    xlabels = np.arange(sumTime-limit_x_1, sumTime + limit_x_2, limit_x_2)
    ax.set_xticklabels(xlabels, Fontsize=6)

    plt.draw()
    if counter == 0:
        plt.pause(30)
    plt.pause(0.7)

    counter = counter + 1

cv2.destroyAllWindows()
plt.show()

testing_velocities = (testing_velocities - velocities_mean) / velocities_std


mean_squared_error = np.mean((np.array(w) - testing_velocities) ** 2)
print(mean_squared_error)

r_score = r2_score(testing_velocities, np.array(w))
print(r_score)
