import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from torch import zeros, FloatTensor
import numpy as np
import time
import cv2


def CNNParameters(algorithm, layer_index):
    if algorithm == "ResNet18":
        layers = np.array(["conv1",
                           "layer1.0.conv1", "layer1.0.conv2", "layer1.1.conv1", "layer1.1.conv1", #4
                           "layer2.0.conv1", "layer2.0.conv2", "layer2.1.conv1", "layer2.1.conv2", #8
                           "layer3.0.conv1", "layer3.0.conv2", "layer3.1.conv1", "layer3.1.conv2", #12
                           "layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2"]) #16

        filters = np.array([112, 56, 56, 56, 56, 28, 28, 28, 28, 14, 14, 14, 14, 7, 7, 7, 7])
        depth_of_filters = np.array([64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512])
    else:
        layers, filters, depth_of_filters = [], [], []

    return layers[layer_index], filters[layer_index], depth_of_filters[layer_index]


def CNN(cnn_name, layer_index, images_paths, dataset_name):
    image_width = 224
    image_height = 224
    number_of_images = len(images_paths)

    layer_name, filters, depth_of_filters = CNNParameters(cnn_name, layer_index)

    # Load the pretrained model
    model = models.resnet18(pretrained=True)
    print(model)
    # Use the model object to select the desired layer

    splitted_layer_name = layer_name.split(".")
    if len(splitted_layer_name) == 1:
        layer = model._modules.get(splitted_layer_name[0])
    else:
        layer = model._modules.get(splitted_layer_name[0]).\
                _modules.get(splitted_layer_name[1])._modules.get(splitted_layer_name[2])

    # Set model to evaluation mode
    model.eval()

    dimension_of_features = filters**2 * depth_of_filters
    descriptors = np.empty((number_of_images, dimension_of_features))
    images_labels = []
    elapsed_time = np.empty((number_of_images, 1))
    for ii, image_path in enumerate(images_paths):
        print(ii)
        image_name = image_path.split("\\")[-1].split(".")[0]
        images_labels.append(image_name)
        start = time.time()
        descriptors[ii, :] = CNN_Feature_Extraction(image_path, model, layer, dimension_of_features)
        elapsed_time[ii] = time.time() - start
    # return descriptors, images_labels
    descriptor_file = "_".join([dataset_name, cnn_name, layer_name.replace(".", "_"), "descriptor"])
    np.savez(descriptor_file, descriptors=descriptors, dataset_name=dataset_name, image_height=image_height,
             image_width=image_width, cnn_name=cnn_name, layer_name=layer_name, number_of_images=number_of_images,
             images_labels=images_labels, elapsed_time=elapsed_time)


def CNN_Feature_Extraction(image_path, model, layer, dimension_of_features):
    # 1. Load the image with CV2 library
    img = cv2.imread(image_path)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = image_loader(img)
    # 3. Create a vector of zeros that will hold our feature vector
    deep_features = zeros(dimension_of_features)

    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        deep_features.copy_(o.data.flatten())
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return deep_features


def image_loader(image):
    loader = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
    image = loader(image)
    image = FloatTensor(image)
    image = image.unsqueeze(0)
    image = Variable(image, requires_grad=True)

    return image
