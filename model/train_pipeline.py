import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from model.model import UNet
from utils.scheduler import DDPMScheduler
from utils.sampler import ddpm_sample

class DiffusionLightning(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_cfg = cfg["training"]

        # DDPM Scheduler
        self.scheduler = DDPMScheduler(num_timesteps=self.train_cfg["num_timesteps"])

        # Model
        self.model = UNet(cfg)

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
        self.log("train_loss", loss, prog_bar=True, on_epoch = True)
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

    def on_validation_epoch_end(self):
        # Generate and log sample images every few epochs
        with torch.no_grad():
            samples = self.sample(n=4)  # Generate 4 sample images
            
            # Log images to TensorBoard
            self.logger.experiment.add_images(
                "generated_samples", 
                samples, 
                self.current_epoch,
                dataformats='NCHW'
            )

    # ── Optimizer + Scheduler ─────────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg["learning_rate"],)
        return optimizer

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