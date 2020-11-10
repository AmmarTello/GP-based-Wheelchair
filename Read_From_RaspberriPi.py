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


exp = 4 #4 # 3
end_value = 70 #70  # 14
with np.load("exp_" + str(exp) + ".npz") as results:
    w = results["Y_StarMean"]
    end = results["test_elapsed_time"]
    training_velocities = results["training_velocities"]
    velocities_mean = results["velocities_mean"]
    velocities_std = results["velocities_std"]

ground_truth_velocities_file_name = "Ground Truth.csv"  # Velocities
with open(ground_truth_velocities_file_name) as csv_file:
    velocities_file = csv.reader(csv_file, delimiter=',')
    velocities_file_content = np.array([[int(row[0].split(".")[0]), float(row[1])] for row in velocities_file])

dataset_path = "..\\Dataset\\Campus\\"
number_of_images, images_paths, images_labels = collect_images_paths(dataset_path + str(exp) + "\\", "png")

sorter = np.argsort(velocities_file_content[:, 0])
labels_indices = sorter[np.searchsorted(velocities_file_content[:, 0], images_labels, sorter=sorter)]
testing_velocities = velocities_file_content[labels_indices, 1]
testing_velocities = (testing_velocities - velocities_mean) / velocities_std


end = np.append(end, 0)/2
end = np.cumsum(end) - end[0]
# end = np.insert(end, 0, 0)
# w = np.insert(w, 0, 0)

w = w[:end_value]
end = end[:end_value]
print(np.mean(end))

w = np.array(w)
w = (w - velocities_mean) / velocities_std


fig, ax = plt.subplots()
ax.plot(end, w, marker=".", linewidth=4, markersize=18, markerfacecolor="tab:red", markeredgecolor="tab:red")
ax.set_xlim([0, 1.5])
ax.set_ylim([-4, 4])
majorX_ticks = np.arange(-3.5, 4, 0.5)
minorX_ticks = np.arange(-3.5, 4, 0.5)


# major_ticks = np.arange(0, 1.5, 0.25)
# minor_ticks = np.arange(0, 1.5, 0.25)
major_ticks = np.arange(0, 7.1, 0.5)
minor_ticks = np.arange(0, 7.1, 0.5)

xlabels = np.arange(0, 8, 0.5)
# xlabels = np.arange(0, 1.6, 0.25)
ax.set_xticks(major_ticks)
ax.set_xticklabels(xlabels, Fontsize=18)

ax.set_xticks(minor_ticks, minor=True)

ylabels = np.arange(-3.5, 4.1, 0.5)
ax.set_yticks(majorX_ticks)
ax.set_yticklabels(ylabels, Fontsize=18)

ax.set_yticks(minorX_ticks, minor=True)


plt.title("Corridor Following with Multiple Interventions", fontsize=24)
# plt.title("Wheelchair initiates from angle $\\theta$", fontsize=24)
plt.xlabel("Time (s)", fontsize=24)
plt.ylabel("$\omega$ (rad/s)", fontsize=24)
plt.show()

mean_squared_error = np.mean((w - testing_velocities[:end_value]) ** 2)
print(mean_squared_error)

r_score = r2_score(testing_velocities[:end_value], w)
print(r_score)

# saved_file_name = "exp_" + str(exp) + ".npz"
# np.savez(saved_file_name, exp=exp, L=L, Sigma=Sigma, K=K,
#          C=C, InvC=InvC, Y_StarMean=w,
#          test_elapsed_time=end,
#          training_labels=training_labels,
#          training_velocities=training_velocities,
#          velocities_mean=velocities_mean, velocities_std=velocities_std)
