import torch
import random


class RandomFlip3D:

    def __call__(self, image, mask):

        if random.random() > 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])

        return image, mask


class IntensityScale:

    def __call__(self, image, mask):

        scale = 0.9 + 0.2 * torch.rand(1)

        image = image * scale

        return image, mask