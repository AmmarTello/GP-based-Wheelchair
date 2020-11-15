import time
import cv2
from HOGFunc import hog_parameters, extended_hog_parameters
from GP_New import Predict
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
from basic_subertoorh_instructions import initialization, send_vel, stop, send_vel_2


image_height = 224
image_width = 224
fileName = "GP_train_Normal_Noisy_test_Normal_Noisy_Hog_9_32_32_32.npz"
normal_descriptor_name = "Normal_Hog_9_32_32_32_descriptor.npz"
noisy_descriptor_name = "Noisy_Hog_9_32_32_32_descriptor.npz"

with np.load(fileName) as results:
    L = results["L"]
    Sigma = results["Sigma"]
    K = results["K"]
    C = results["C"]
    InvC = results["InvC"]
    training_velocities = results["training_velocities"]
    training_labels = results["training_labels"]

number_of_training_images = len(training_velocities)

with np.load(normal_descriptor_name) as normal_descriptor_file:
    normal_descriptor = normal_descriptor_file["descriptors"]
    normal_ground_truth_labels = normal_descriptor_file["images_labels"]
with np.load(noisy_descriptor_name) as noisy_descriptor_file:
    noisy_descriptor = noisy_descriptor_file["descriptors"]
    noisy_ground_truth_labels = noisy_descriptor_file["images_labels"]

max_vel = np.max(training_velocities)
min_vel = np.min(training_velocities)

velocities_mean = np.mean(training_velocities)
velocities_std = np.std(training_velocities)
training_velocities = (training_velocities - velocities_mean) / velocities_std



ground_truth_labels = np.concatenate([normal_ground_truth_labels, noisy_ground_truth_labels])

training_descriptors = np.concatenate([normal_descriptor, noisy_descriptor], 0)
training_descriptors = np.array([training_descriptors[ground_truth_labels == str(label), :] for label in training_labels]).squeeze().T

features_per_cell, block_size, block_stride, cell_size = hog_parameters()
derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, \
nlevels, signedGradients, winSize = extended_hog_parameters(image_height, image_width)

hog = cv2.HOGDescriptor(winSize, block_size, block_stride, cell_size, features_per_cell, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)

InvCxY = np.dot(InvC, training_velocities)

sabertooth_object = initialization()

camera = PiCamera()
camera.resolution = (image_width, image_height)
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=(image_width, image_height))  #640, 480
time.sleep(0.1)
end = 0
counter = 1
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    start_cycle = time.time()
    cv2.imshow("Frame", image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(image, 3)  # Removes salt and pepper noise by convolving the image with a (3,3) square kernel
    image = cv2.GaussianBlur(image, (3, 3), 0)

    rawCapture.truncate(0)

    descriptor = hog.compute(image, winSize)

    Y_StarMean = Predict(training_descriptors, descriptor, L, InvCxY)

    if cv2.waitKey(1) & 0xFF == ord("s"):
        stop(sabertooth_object)
        break

    Y_StarMean = Y_StarMean * velocities_std + velocities_mean
    # send_vel(Y_StarMean, sabertooth_object)

    send_vel_2(Y_StarMean, sabertooth_object, max_vel, min_vel)
    print(Y_StarMean)
    end += time.time() - start_cycle
    print("Time = ", time.time() - start_cycle)

    counter = counter + 1


cv2.destroyAllWindows()
