import random
import torch

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)

        return image


class RandomRotate(object):
    def __call__(self, image):
        angle = random.randint(0, 3)
        if type(image) is not list:
            image = torch.rot90(image, angle, [1, 2])
        else:
            for i in range(len(image)):
                image[i] = torch.rot90(image[i], angle, [1, 2])
        
        return image


class RandomFlip(object):
    def __call__(self, image):
        if random.randint(0, 1):
            if type(image) is not list:
                channel, height, width = image.shape
                image = image[:, :, torch.arange(width-1, -1, -1)]
            else:
                for i in range(len(image)):
                    channel, height, width = image[i].shape
                    image[i] = image[i][:, :, torch.arange(width-1, -1, -1)]

        return image