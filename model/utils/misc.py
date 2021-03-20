from collections import OrderedDict
import numpy as np
from PIL import Image
import os
import cv2

def str2bool(s):
    return s.lower() in ('true', '1')

def fix_model_state_dict(state_dict):
    # remove 'module.' of dataparallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


def flow2color(flow):
    height, width, _ = flow.shape
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag = mag*10/ 1.4128270298619277
    # print(mag.min(), mag.max())
    # mag = mag*255 / (1.4128270298619277 * 4)
    # mag = mag*255 / 6.0094142822647285
    hsv[..., 2] = np.clip(mag, 0, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr


class SaveTorchImage(object):
    def __init__(self, cfg):
        self.transforms = Compose([
            ToNumpy(),
            Denormalize(),
            ConvertToInts(),
        ])

    def __call__(self, image, save_path):
        image = self.transforms(image)
        image = image[:, :, ::-1]
        image = Image.fromarray(image)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        
        return image

class ToNumpy(object):
    def __call__(self, image):
        return image.numpy().transpose((1, 2, 0))

class Denormalize(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image *= self.std
        image += self.mean
        image *= 255

        return image

class ConvertToInts(object):
    def __call__(self, image):
        return np.clip(image, 0, 255).astype(np.uint8)
