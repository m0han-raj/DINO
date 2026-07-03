import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torchvision.datasets import STL10
from src.augmentations import MultiCropAugmentation

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")

ds = STL10(root=DATA_ROOT, split="unlabeled",
           transform=MultiCropAugmentation(), download=True)
crops = ds[0][0]
print(len(crops))            # 6
print(crops[0].shape)        # torch.Size([3, 64, 64])
print(crops[2].shape)        # torch.Size([3, 32, 32])