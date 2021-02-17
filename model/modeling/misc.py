import torch
import torch.nn as nn

class Nearest(object):
    def __init__(self):
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def __call__(self, x):
        batch, frame, channel, width, height = x.shape

        out = torch.empty((batch, frame, 3, width*2, height*2)).to(x.device)
        
        out[:, :, 0, :, :] = self.up(x[:, :, 0, :, :])
        out[:, :, 1, :, :] = ( self.up(x[:, :, 1, :, :]) + self.up(x[:, :, 2, :, :]) )/ 2
        out[:, :, 2, :, :] = self.up(x[:, :, 3, :, :])

        return out
    
class RGGB2channel(object):
    def __init__(self):
        pass
    
    def __call__(self, x):
        return x
