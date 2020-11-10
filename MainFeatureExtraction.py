from Image_functions import collect_images_paths
from HOGFunc import hog_func
from CNN import CNN

features_extraction_number = 1
all_features_methods = ["Raw", "Hog", "ResNet18"]
features_extraction_method = all_features_methods[features_extraction_number]
dataset_name = "Normal"

layerIndex = 12

base_path = "H:\\Desktop\\CARA\\Dataset\\"
image_extension = "png"
image_width, image_height = (224, 224)

number_of_images, images_paths, images_labels = collect_images_paths(base_path + dataset_name + "\\", image_extension)

if features_extraction_method == "Hog":
    hog_func(images_paths, image_height, image_width, dataset_name)
elif features_extraction_method == "ResNet18":
    CNN(features_extraction_method, layerIndex, images_paths, dataset_name)
