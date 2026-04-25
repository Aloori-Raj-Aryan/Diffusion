import torch
from pathlib import Path
import logging
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, random_split
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

class FlatImageDataset(torch.utils.data.Dataset):
    """Loads all images from a flat directory (no class subfolders needed)."""

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.paths = [
            p for p in Path(root).rglob("*")
            if p.suffix.lower() in VALID_EXTENSIONS
        ]
        if not self.paths:
            raise FileNotFoundError(f"No images found in: {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0   # dummy label — keeps DataLoader interface consistent


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    paths = cfg["paths"]
    train_cfg = cfg["training"]

    transform = transforms.Compose([
        transforms.Resize((train_cfg["image_size"], train_cfg["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),   # → [-1, 1]
    ])

    dataset = FlatImageDataset(root=paths["dataset_path"], transform=transform)

    val_size = int(len(dataset) * train_cfg["val_split"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader

