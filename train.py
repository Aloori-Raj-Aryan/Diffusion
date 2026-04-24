"""
Diffusion Model Training Script
Compatible with config.yaml
"""

import os
import math
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import numpy as np
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Noise Scheduler (DDPM)
# ─────────────────────────────────────────────────────────────────────────────
class DDPMScheduler:
    """Linear beta schedule as in Ho et al. 2020."""

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # For q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t from q(x_t | x_0)."""
        noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_t, noise


# ─────────────────────────────────────────────────────────────────────────────
# U-Net Building Blocks
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch * 2))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, -1).transpose(1, 2)
        h, _ = self.attn(h, h, h)
        return x + h.transpose(1, 2).view(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    U-Net for noise prediction in DDPM.

    Args:
        image_size:      Spatial resolution of input images.
        in_channels:     Number of image channels (e.g. 3 for RGB).
        model_channels:  Base channel width; doubled at each down-scale.
        num_res_blocks:  ResBlocks per resolution level.
        attn_resolutions: Resolutions at which self-attention is applied.
        dropout:         Dropout probability inside ResBlocks.
    """

    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (8, 16),
        dropout: float = 0.1,
    ):
        super().__init__()
        ch = model_channels
        time_emb_dim = ch * 4

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(ch),
            nn.Linear(ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Channel schedule: [ch, ch*2, ch*4]
        ch_mult = [1, 2, 4]
        channels = [ch * m for m in ch_mult]

        # ── Encoder ───────────────────────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.down_attns = nn.ModuleList()

        cur_res = image_size
        prev_ch = ch
        for out_ch in channels:
            blocks = nn.ModuleList(
                [ResBlock(prev_ch if i == 0 else out_ch, out_ch, time_emb_dim, dropout)
                 for i in range(num_res_blocks)]
            )
            self.down_blocks.append(blocks)
            self.down_attns.append(
                AttentionBlock(out_ch) if cur_res in attn_resolutions else nn.Identity()
            )
            self.down_samples.append(
                Downsample(out_ch) if out_ch != channels[-1] else nn.Identity()
            )
            prev_ch = out_ch
            if out_ch != channels[-1]:
                cur_res //= 2

        # ── Bottleneck ────────────────────────────────────────────────────────
        bot_ch = channels[-1]
        self.mid_block1 = ResBlock(bot_ch, bot_ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(bot_ch)
        self.mid_block2 = ResBlock(bot_ch, bot_ch, time_emb_dim, dropout)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_attns = nn.ModuleList()

        rev_channels = list(reversed(channels))
        for i, out_ch in enumerate(rev_channels):
            in_ch = rev_channels[i - 1] if i > 0 else bot_ch
            skip_ch = out_ch  # from encoder skip connection
            blocks = nn.ModuleList(
                [ResBlock(in_ch + skip_ch if i2 == 0 else out_ch, out_ch, time_emb_dim, dropout)
                 for i2 in range(num_res_blocks)]
            )
            self.up_blocks.append(blocks)
            self.up_attns.append(
                AttentionBlock(out_ch) if cur_res in attn_resolutions else nn.Identity()
            )
            self.up_samples.append(
                Upsample(out_ch) if i < len(rev_channels) - 1 else nn.Identity()
            )
            if i < len(rev_channels) - 1:
                cur_res *= 2

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        h = self.input_conv(x)
        skips = [h]

        # Encoder
        for blocks, attn, down in zip(self.down_blocks, self.down_attns, self.down_samples):
            for block in blocks:
                h = block(h, t_emb)
            h = attn(h) if isinstance(attn, AttentionBlock) else h
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for blocks, attn, up in zip(self.up_blocks, self.up_attns, self.up_samples):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)
            h = attn(h) if isinstance(attn, AttentionBlock) else h
            h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
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
        logger.info(f"Found {len(self.paths):,} images in {root}")

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

    logger.info(f"Dataset: {len(train_ds):,} train  |  {len(val_ds):,} val")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class DiffusionTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.train_cfg = cfg["training"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Scheduler
        self.scheduler = DDPMScheduler(
            num_timesteps=self.train_cfg["num_timesteps"]
        ).to(self.device)

        # Model
        self.model = UNet(
            image_size=self.train_cfg["image_size"],
            model_channels=self.train_cfg["model_channels"],
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {num_params:,}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )

        # Cosine LR schedule
        self.scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.train_cfg["epochs"]
        )

        # Output dirs
        self.run_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.sample_dir = self.run_dir / "samples"
        for d in [self.ckpt_dir, self.sample_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_val_loss = float("inf")

    # ── Training step ─────────────────────────────────────────────────────────
    def train_step(self, x0: torch.Tensor) -> float:
        x0 = x0.to(self.device)
        t = torch.randint(0, self.train_cfg["num_timesteps"], (x0.size(0),), device=self.device)

        x_t, noise = self.scheduler.add_noise(x0, t)
        pred_noise = self.model(x_t, t)
        loss = F.mse_loss(pred_noise, noise)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg["max_grad_norm"])
        self.optimizer.step()

        return loss.item()

    # ── Validation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for x0, _ in tqdm(val_loader, desc="  Validation", leave=False):
            x0 = x0.to(self.device)
            t = torch.randint(0, self.train_cfg["num_timesteps"], (x0.size(0),), device=self.device)
            x_t, noise = self.scheduler.add_noise(x0, t)
            pred_noise = self.model(x_t, t)
            total_loss += F.mse_loss(pred_noise, noise).item()
        self.model.train()
        return total_loss / len(val_loader)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = ""):
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.cfg,
        }
        suffix = f"_{tag}" if tag else f"_epoch{epoch:04d}"
        path = self.ckpt_dir / f"ckpt{suffix}.pt"
        torch.save(ckpt, path)
        logger.info(f"  Saved checkpoint → {path}")
        return path

    # ── Sample grid ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def save_samples(self, epoch: int, n: int = 16):
        self.model.eval()
        samples = ddpm_sample(
            self.model,
            self.scheduler,
            shape=(n, 3, self.train_cfg["image_size"], self.train_cfg["image_size"]),
            device=self.device,
            num_steps=self.train_cfg["num_timesteps"],
        )
        grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
        path = self.sample_dir / f"epoch_{epoch:04d}.png"
        save_image(grid, path)
        self.model.train()
        logger.info(f"  Saved samples    → {path}")

    # ── Main training loop ────────────────────────────────────────────────────
    def fit(self):
        train_loader, val_loader = build_dataloaders(self.cfg)
        epochs = self.train_cfg["epochs"]

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch:>4}/{epochs}", leave=False)

            for batch_idx, (x0, _) in enumerate(pbar):
                loss = self.train_step(x0)
                epoch_loss += loss
                self.global_step += 1

                if self.global_step % self.train_cfg["log_interval"] == 0:
                    pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{self.scheduler_lr.get_last_lr()[0]:.2e}")

            epoch_loss /= len(train_loader)
            self.scheduler_lr.step()

            # Validation
            val_loss = None
            if epoch % self.train_cfg["val_interval"] == 0:
                val_loss = self.validate(val_loader)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, tag="best")
                logger.info(
                    f"Epoch {epoch:4d} | train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f} | best={self.best_val_loss:.4f}"
                )
            else:
                logger.info(f"Epoch {epoch:4d} | train_loss={epoch_loss:.4f}")

            # Periodic checkpoint & samples
            if epoch % self.train_cfg["save_interval"] == 0:
                self.save_checkpoint(epoch, val_loss or epoch_loss)
                self.save_samples(epoch)

        # Final save
        self.save_checkpoint(epochs, self.best_val_loss, tag="final")
        logger.info("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# DDPM Sampler (shared with inference.py)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    scheduler: DDPMScheduler,
    shape: tuple,
    device: torch.device,
    num_steps: int = 1000,
) -> torch.Tensor:
    x = torch.randn(shape, device=device)
    timesteps = list(range(num_steps - 1, -1, -1))

    for t_val in tqdm(timesteps, desc="Sampling", leave=False):
        t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
        pred_noise = model(x, t)

        alpha = scheduler.alphas[t_val]
        alpha_bar = scheduler.alphas_cumprod[t_val]
        beta = scheduler.betas[t_val]

        # Predict x0
        x0_pred = (x - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        # Posterior mean
        mean = (alpha.sqrt() * (1 - scheduler.alphas_cumprod_prev[t_val]) * x
                + scheduler.alphas_cumprod_prev[t_val].sqrt() * beta * x0_pred) \
               / (1 - alpha_bar)

        if t_val > 0:
            noise = torch.randn_like(x)
            var = scheduler.posterior_variance[t_val].clamp(min=1e-20)
            x = mean + var.sqrt() * noise
        else:
            x = mean

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train a DDPM diffusion model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trainer = DiffusionTrainer(cfg)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(ckpt["model_state"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        trainer.global_step = ckpt["global_step"]
        logger.info(f"Resumed from checkpoint: {args.resume} (epoch {ckpt['epoch']})")

    trainer.fit()


if __name__ == "__main__":
    main()