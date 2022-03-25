import os.path
from glob import glob

from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    General dataset class that responsible of gathering images and apply the required transforms into the images.
    """
    def __init__(self, dataset_roots, csv_root, transform=None, image_extension="png", read_image_flag=True):
        """
        Args:
            dataset_roots (list): list of datasets roots
            csv_root (str): the path of the csv file that contains two columns: "image": contains names of images.
                            "w": contains the ground truth velocities
            transform (Transform): which is an object of the Transform class to be applied into the images.
            image_extension (str): the images extension
            read_image_flag (bool): whether to read the images (True) or just gathering the images paths with the
                                    corresponding velocities (False).
        """
        super().__init__()

        self.dataset_roots = dataset_roots
        self.csv_root = csv_root
        self.transform = transform
        self.image_extension = image_extension
        self.read_image_flag = read_image_flag

        self.images_paths = []
        for root in self.dataset_roots:
            self.images_paths.extend(glob(os.path.join(root, f"*.{image_extension}")))

        self.dataset_df = pd.read_csv(csv_root)

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        image_name = os.path.split(image_path)[-1]
        velocity = self.dataset_df.loc[self.dataset_df["image"] == image_name, "w"].values[0]

        image = image_path
        if self.read_image_flag:
            image = self.read_image(image_path)

        return image, velocity, image_name

    def read_image(self, image_path):
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.dataset_df["image"])
