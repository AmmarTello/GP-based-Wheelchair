import torchvision.models as models
import torch.nn as nn

from . import DescriptorInterface
from configuration import Descriptor


class ResNet18(nn.Module, DescriptorInterface):
    """
    The ResNet18 model as descriptor
    """
    DESCRIPTOR_ALGORITHM = Descriptor.RESNET18

    def __init__(self, layer, pretrained=True, evaluation=True):
        super().__init__()
        self.extractor = models.resnet18(pretrained=pretrained)
        if evaluation:
            self.extractor.eval()

        self.layer = layer

    def extract(self, image):
        return self.forward(image)

    def forward(self, image):
        features = []

        def hook_fn(m, i, o):
            features.append(o.data.flatten().numpy())

        layer = eval(f"self.extractor.{self.layer}")
        h = layer.register_forward_hook(hook_fn)

        self.extractor(image[None, ...])

        h.remove()

        return features[0]

    def get_descriptor_details(self):
        return self.layer.replace("[", "_").replace("]", "_").replace(".", "")




