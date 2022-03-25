import os
import numpy as np
from sklearn.model_selection import train_test_split

from data_preparation.dataset import ImageDataset
from utilities.constants import ImageExtension, Subset


class DatasetSplitting:
    """
    This class is responsible of splitting the dataset into train and test splits.
    """
    def __init__(self, root, train_datasets_paths, test_dataset, test_subset,
                 ground_truth_velocities_file_name, test_percent, seed):
        self.root = root
        self.train_datasets_paths = train_datasets_paths
        self.test_dataset = test_dataset
        self.test_subset = test_subset
        self.ground_truth_velocities_file_name = ground_truth_velocities_file_name
        self.test_percent = test_percent
        self.seed = seed

    def __call__(self):
        image_extension = ImageExtension.PNG
        dataset = ImageDataset(self.train_datasets_paths, self.ground_truth_velocities_file_name,
                               image_extension=image_extension, read_image_flag=False)

        images_paths, velocities, images_names = [], [], []
        for img_path, velocity, img_label in dataset:
            images_paths.append(img_path)
            velocities.append(velocity)
            images_names.append(img_label)

        if self.test_dataset is None:
            train_images_paths, test_images_paths, \
                train_images_names, test_images_names, \
                train_velocities, test_velocities = \
                train_test_split(images_paths, images_names, velocities,
                                 test_size=self.test_percent, random_state=self.seed)

        else:
            if self.test_subset == Subset.JPEG_COMPRESSION:
                image_extension = ImageExtension.JPEG

            datasets_paths = [os.path.join(self.root, self.test_dataset, self.test_subset)]
            dataset = ImageDataset(datasets_paths, self.ground_truth_velocities_file_name,
                                   image_extension=image_extension, read_image_flag=False)

            subset_images_paths, subset_velocities, subset_images_names = [], [], []
            for img_path, velocity, img_label in dataset:
                subset_images_paths.append(img_path)
                subset_velocities.append(velocity)
                subset_images_names.append(img_label)

            number_of_testing_images = round(self.test_percent * len(subset_images_names))
            test_percent = number_of_testing_images / len(images_names)
            _, test_images_paths, _, test_images_names, _, test_velocities = \
                train_test_split(subset_images_paths, subset_images_names, subset_velocities,
                                 test_size=test_percent, random_state=self.seed)
            test_indices = np.invert(np.isin(images_names, test_images_names))
            train_images_paths = np.array(images_paths)[test_indices]
            train_images_names = np.array(images_names)[test_indices]
            train_velocities = np.array(velocities)[test_indices]

        train_velocities_mean = np.mean(train_velocities)
        train_velocities_std = np.std(train_velocities)
        train_velocities = (train_velocities - train_velocities_mean) / train_velocities_std
        test_velocities = (test_velocities - train_velocities_mean) / train_velocities_std

        return {"train_images_paths": train_images_paths,
                "train_images_names": train_images_names,
                "train_velocities": train_velocities,
                "train_velocities_mean": train_velocities_mean,
                "train_velocities_std": train_velocities_std},\
               {"test_images_paths": test_images_paths,
                "test_images_names": np.array(test_images_names),
                "test_velocities": test_velocities}


