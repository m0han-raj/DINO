"""Multi-crop augmentation pipeline for DINO (CPU edition).
2 global crops (64x64) + 4 local crops (32x32)."""
import torch
from torchvision import transforms


class MultiCropAugmentation:
    def __init__(self, global_size=64, local_size=32, n_local=4):
        flip_and_color = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066),   # STL-10 stats
                                 (0.2603, 0.2566, 0.2713)),
        ])

        # Global crops: cover 40-100% of the image
        self.global_t = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
            flip_and_color,
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
            normalize,
        ])
        # Local crops: cover 5-40% of the image
        self.local_t = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.4)),
            flip_and_color,
            normalize,
        ])
        self.n_local = n_local

    def __call__(self, img):
        crops = [self.global_t(img), self.global_t(img)]
        crops += [self.local_t(img) for _ in range(self.n_local)]
        return crops   # list of 6 tensors