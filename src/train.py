"""DINO training loop — CPU edition.
Run: python -m src.train"""
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
from tqdm import tqdm

from src.vision_transformer import VisionTransformer
from src.dino_head import DINOHead
from src.dino_loss import DINOLoss
from src.augmentations import MultiCropAugmentation


# ---------------- CPU settings ----------------
torch.set_num_threads(os.cpu_count())
DEVICE = "cpu"

# ---------------- Hyperparameters ----------------
EPOCHS = 20
BATCH_SIZE = 32
ACCUM_STEPS = 4            # effective batch = 128
LR = 5e-4
WEIGHT_DECAY = 0.04
EMA_BASE = 0.996
N_IMAGES = 5000            # subset of STL-10 unlabeled
N_CROPS = 6                # 2 global + 4 local
CKPT_DIR = "outputs/checkpoints"


class MultiCropWrapper(nn.Module):
    """Backbone + head, handling crops of different resolutions.
    Groups crops by resolution so same-size crops share one forward pass."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, crops):
        # crops: list of tensors, possibly different HxW
        sizes = torch.tensor([c.shape[-1] for c in crops])
        # indices where resolution changes (e.g. [2, 6])
        idx_splits = torch.cumsum(torch.unique_consecutive(
            sizes, return_counts=True)[1], 0)
        out, start = [], 0
        for end in idx_splits:
            batch = torch.cat(crops[start:end])   # same-size crops together
            out.append(self.backbone(batch))
            start = end
        return self.head(torch.cat(out))


def ema_momentum(step, total_steps):
    """Cosine schedule 0.996 -> 1.0 over training."""
    return 1 - (1 - EMA_BASE) * (math.cos(math.pi * step / total_steps) + 1) / 2


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    ds = STL10(root="data", split="unlabeled",
               transform=MultiCropAugmentation(), download=True)
    ds = Subset(ds, range(N_IMAGES))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=False, drop_last=True)

    student = MultiCropWrapper(VisionTransformer(), DINOHead()).to(DEVICE)
    teacher = MultiCropWrapper(VisionTransformer(), DINOHead()).to(DEVICE)
    teacher.load_state_dict(student.state_dict())   # identical start
    for p in teacher.parameters():
        p.requires_grad = False                      # teacher never has grads

    dino_loss = DINOLoss(n_crops=N_CROPS).to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(loader)
    step = 0

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{EPOCHS}")
        running = 0.0
        optimizer.zero_grad()
        for i, (crops, _) in enumerate(pbar):
            crops = [c.to(DEVICE) for c in crops]

            with torch.no_grad():
                teacher_out = teacher(crops[:2])     # global crops only
            student_out = student(crops)             # all 6 crops

            loss = dino_loss(student_out, teacher_out) / ACCUM_STEPS
            loss.backward()
            running += loss.item() * ACCUM_STEPS

            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
                optimizer.step()
                optimizer.zero_grad()

                # EMA teacher update
                m = ema_momentum(step, total_steps)
                with torch.no_grad():
                    for ps, pt in zip(student.parameters(),
                                      teacher.parameters()):
                        pt.mul_(m).add_(ps, alpha=1 - m)
                step += 1

            pbar.set_postfix(loss=f"{running / (i + 1):.3f}")

        torch.save({"student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "epoch": epoch},
                   f"{CKPT_DIR}/dino_epoch{epoch+1}.pt")


if __name__ == "__main__":
    main()