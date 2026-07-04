import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision_transformer import VisionTransformer
import torch
v = VisionTransformer()
print(v(torch.randn(2, 3, 64, 64)).shape)   # (2, 192)
print(v(torch.randn(2, 3, 32, 32)).shape)   # (2, 192)  <- interp works
print(v.get_last_attention(torch.randn(2, 3, 64, 64)).shape)  # (2, 3, 65, 65)
print(sum(p.numel() for p in v.parameters()) / 1e6, "M params")  # ~2.7