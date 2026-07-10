"""Freeze the trained backbone, train ONLY a linear classifier on STL-10
labeled data. Run: python -m src.evaluate --ckpt outputs/checkpoints/dino_epoch20.pt"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import STL10

from src.vision_transformer import VisionTransformer


def get_loader(split, batch=64):
    t = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066),
                             (0.2603, 0.2566, 0.2713))])
    ds = STL10(root="data", split=split, transform=t, download=True)
    return DataLoader(ds, batch_size=batch, shuffle=(split == "train"))


def extract_features(backbone, loader):
    feats, labels = [], []
    backbone.eval()
    with torch.no_grad():
        for x, y in loader:
            feats.append(backbone(x))
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def main(ckpt_path):
    backbone = VisionTransformer()
    state = torch.load(ckpt_path, map_location="cpu")
    # teacher backbone gives slightly better features than student
    bb_state = {k.replace("backbone.", ""): v
                for k, v in state["teacher"].items()
                if k.startswith("backbone.")}
    backbone.load_state_dict(bb_state)

    train_f, train_y = extract_features(backbone, get_loader("train"))
    test_f, test_y = extract_features(backbone, get_loader("test"))

    clf = nn.Linear(192, 10)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for epoch in range(50):
        idx = torch.randperm(len(train_f))
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]
            loss = nn.functional.cross_entropy(clf(train_f[b]), train_y[b])
            opt.zero_grad(); loss.backward(); opt.step()

    acc = (clf(test_f).argmax(1) == test_y).float().mean().item()
    print(f"Linear probe top-1 accuracy: {acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    main(p.parse_args().ckpt)