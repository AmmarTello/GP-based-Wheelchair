import torchvision.models as models
import torch.nn as nn

from . import FeatureExtractionInterface
from configuration import FeatureExtractionAlgorithms


class ResNet18(nn.Module, FeatureExtractionInterface):
    DESCRIPTOR_ALGORITHM = FeatureExtractionAlgorithms.RESNET18

    def __init__(self, layer, pretrained=True):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.layer = layer

    def extract(self, image):
        return self.forward(image)

    def forward(self, image):
        features = []

        def hook_fn(m, i, o):
            features.append(o.data.flatten())

        layer = eval(f"self.model.{self.layer}")
        h = layer.register_forward_hook(hook_fn)

        self.model(image)

        h.remove()

        return features

    def get_descriptor_details(self):
        return self.layer




