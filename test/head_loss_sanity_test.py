import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.dino_head import DINOHead
from src.dino_loss import DINOLoss

head, loss_fn = DINOHead(), DINOLoss()
B = 4
student = torch.randn(6 * B, 4096, requires_grad=True)
teacher = torch.randn(2 * B, 4096)
loss = loss_fn(student, teacher)
print(loss.item())  # a positive number, roughly ln(4096) ≈ 8.3 at init
loss.backward()
print(student.grad is not None)  # True
