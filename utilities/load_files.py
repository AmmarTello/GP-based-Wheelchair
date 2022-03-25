import numpy as np


class LoadResults:
    """
    Utilities to be used when loading results.
    """
    @staticmethod
    def load_descriptors(datasets, descriptor_algorithm, selected_images, selected_velocities):
        """
        Load descriptors files and return the descriptors that corresponding to specific images
        """
        desc_file = descriptor_algorithm.load(datasets[0])
        descriptors = desc_file["descriptors"]
        images_names = desc_file["images_names"]

        for dataset_index in range(1, len(datasets)):
            desc_file = descriptor_algorithm.load(datasets[dataset_index])

            descriptors = np.concatenate([descriptors, desc_file["descriptors"]])
            images_names = np.concatenate([images_names, desc_file["images_names"]])

        selected_output = np.array([[descriptors[int(np.where(images_names == img_name)[0]), :].reshape(1, -1), velocity] for img_name, velocity in
                                    zip(selected_images, selected_velocities)], dtype=object)

        return np.concatenate(selected_output[:, 0], axis=0), selected_output[:, 1]
