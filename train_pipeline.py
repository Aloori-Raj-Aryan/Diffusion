import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from model.model import UNet
from utils.scheduler import DDPMScheduler
from utils.dataloader import DiffusionDataModule
from utils.sampler import ddpm_sample
from utils.checkpoints import get_checkpoints


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
    
    # Update checkpoint dir in config to include run_dir
    cfg.setdefault("paths", {})["checkpoint_dir"] = str(run_dir / "checkpoints")
    
    # Get all checkpoint callbacks from config
    checkpoint_callbacks = get_checkpoints(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator="auto",
        devices="auto",
        precision=16,  # mixed precision (optional but recommended)
        callbacks=checkpoint_callbacks
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()