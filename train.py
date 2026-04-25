import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from model.model import UNet
from model.scheduler import DDPMScheduler
from utils.dataloader import build_dataloaders

class DiffusionTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.train_cfg = cfg["training"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def validate(self, val_loader) -> float:
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

            # Periodic checkpoint & samples
            if epoch % self.train_cfg["save_interval"] == 0:
                self.save_checkpoint(epoch, val_loss or epoch_loss)
                self.save_samples(epoch)

        # Final save
        self.save_checkpoint(epochs, self.best_val_loss, tag="final")


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

    trainer.fit()


if __name__ == "__main__":
    main()