import numpy as np

import torch
import torchvision.datasets as datasets


class CachedImageFolder(datasets.DatasetFolder):
    def __init__(self, root: str):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )

    def __getitem__(self, index: int):
        """
        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        with np.load(path) as data:
            if torch.rand(1) < 0.5:
                moments = data["moments"]
            else:
                moments = data["moments_flip"]

        return moments, target
