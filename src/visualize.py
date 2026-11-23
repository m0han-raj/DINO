"""Extract CLS-token attention maps — the emergent segmentation.
Run: python -m src.visualize --ckpt outputs/checkpoints/dino_epoch20.pt"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.datasets import STL10

from src.vision_transformer import VisionTransformer

OUT = "outputs/attention_maps"


def main(ckpt_path, n_images=8):
    os.makedirs(OUT, exist_ok=True)
    backbone = VisionTransformer()
    state = torch.load(ckpt_path, map_location="cpu")
    bb_state = {
        k.replace("backbone.", ""): v
        for k, v in state["teacher"].items()
        if k.startswith("backbone.")
    }
    backbone.load_state_dict(bb_state)
    backbone.eval()

    t = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ]
    )
    ds = STL10(root="data", split="test", transform=t, download=True)

    for i in range(n_images):
        img, _ = ds[i]
        with torch.no_grad():
            attn = backbone.get_last_attention(img.unsqueeze(0))
        # CLS token's attention over the 64 patch tokens, per head
        cls_attn = attn[0, :, 0, 1:]  # (heads, 64)
        n_heads = cls_attn.shape[0]
        maps = cls_attn.reshape(n_heads, 8, 8)  # 8x8 patch grid

        fig, axes = plt.subplots(1, n_heads + 1, figsize=(3 * (n_heads + 1), 3))
        unnorm = img * torch.tensor([0.2603, 0.2566, 0.2713]).view(
            3, 1, 1
        ) + torch.tensor([0.4467, 0.4398, 0.4066]).view(3, 1, 1)
        axes[0].imshow(unnorm.permute(1, 2, 0).clamp(0, 1))
        axes[0].set_title("input")
        axes[0].axis("off")
        for h in range(n_heads):
            axes[h + 1].imshow(maps[h], cmap="inferno")
            axes[h + 1].set_title(f"head {h}")
            axes[h + 1].axis("off")
        plt.tight_layout()
        plt.savefig(f"{OUT}/attn_{i}.png", dpi=120)
        plt.close()
    print(f"Saved {n_images} attention maps to {OUT}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    main(p.parse_args().ckpt)
