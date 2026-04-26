import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model.model import UNet
from model.scheduler import DDPMScheduler
from utils.dataloader import build_dataloaders


# ─────────────────────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────────────────────
class DiffusionLightning(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_cfg = cfg["training"]

        # DDPM Scheduler
        self.scheduler = DDPMScheduler(
            num_timesteps=self.train_cfg["num_timesteps"]
        )

        # Model
        self.model = UNet(
            image_size=self.train_cfg["image_size"],
            model_channels=self.train_cfg["model_channels"],
        )

    def setup(self, stage=None):
        self.scheduler = self.scheduler.to(self.device)

    def forward(self, x, t):
        return self.model(x, t)

    # ── Training step ─────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        x0, _ = batch
        t = torch.randint(0, self.train_cfg["num_timesteps"], (x0.size(0),), device=self.device)

        x_t, noise = self.scheduler.add_noise(x0, t)
        pred_noise = self(x_t, t)

        loss = F.mse_loss(pred_noise, noise)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    # ── Validation ────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        x0, _ = batch
        t = torch.randint(0, self.train_cfg["num_timesteps"], (x0.size(0),), device=self.device)

        x_t, noise = self.scheduler.add_noise(x0, t)
        pred_noise = self(x_t, t)

        loss = F.mse_loss(pred_noise, noise)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    # ── Optimizer + Scheduler ─────────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.train_cfg["epochs"]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    # ── Sampling (same logic) ─────────────────────────────────
    @torch.no_grad()
    def sample(self, n=16):
        return ddpm_sample(
            self.model,
            self.scheduler,
            shape=(n, 3, self.train_cfg["image_size"], self.train_cfg["image_size"]),
            device=self.device,
            num_steps=self.train_cfg["num_timesteps"],
        )


# ─────────────────────────────────────────────────────────────
# DataModule (cleaner data handling)
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# DDPM Sampler (same as yours)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def ddpm_sample(model, scheduler, shape, device, num_steps=1000):
    x = torch.randn(shape, device=device)

    for t_val in reversed(range(num_steps)):
        t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
        pred_noise = model(x, t)

        alpha = scheduler.alphas[t_val]
        alpha_bar = scheduler.alphas_cumprod[t_val]
        beta = scheduler.betas[t_val]

        x0_pred = (x - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

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


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = DiffusionLightning(cfg)
    data_module = DiffusionDataModule(cfg)

    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="best",
    )

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator="auto",
        devices="auto",
        precision=16,  # mixed precision (optional but recommended)
        callbacks=[checkpoint_callback],
        log_every_n_steps=cfg["training"]["log_interval"],
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()