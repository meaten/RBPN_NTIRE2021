from torchvision.transforms import ToTensor

from .transforms import *


class SyntheticTransforms(object):
    def __init__(self):
        self.transform = Compose([
            ToTensor(),
            RandomRotate(),
            RandomFlip(),
        ])

    def __call__(self, image):
        return self.transform(image)


class RealTransforms(object):
    def __init__(self):
        self.transform = Compose([
            RandomRotate(),
            RandomFlip(),
        ])

    def __call__(self, image):
        return self.transform(image)
