import os
import torch
from pathlib import Path
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.io import ImageReadMode

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

class FlatImageDataset(torch.utils.data.Dataset):
    """Loads all images from a flat directory (no class subfolders needed)."""

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.paths = [
            str(p) for p in Path(root).rglob("*")
            if p.suffix.lower() in VALID_EXTENSIONS
        ]
        if not self.paths:
            raise FileNotFoundError(f"No images found in: {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img =  torchvision.io.read_image(self.paths[idx], mode=ImageReadMode.RGB).float()
        if self.transform:
            img = self.transform(img)
        return img, 0   # dummy label — keeps DataLoader interface consistent


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    paths = cfg["paths"]
    train_cfg = cfg["training"]

    transform = transforms.Compose([
        transforms.Resize((train_cfg["image_size"], train_cfg["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([127.5]*3, [127.5]*3),   # → [-1, 1]
    ])

    dataset = FlatImageDataset(root=paths["dataset_path"], transform=transform)

    val_size = int(len(dataset) * train_cfg["val_split"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    cpu_count = os.cpu_count() or 4
    train_workers = min(4, max(1, cpu_count // 2))
    val_workers = max(1, train_workers // 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
    )
    return train_loader, val_loader

class DiffusionDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_loader, self.val_loader = build_dataloaders(self.cfg)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader