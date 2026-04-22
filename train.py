import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from ema_pytorch import EMA


# ==================== Diffusion Model Architecture ====================

class DiffusionScheduler:
    """Linear noise schedule for diffusion process"""
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For posterior
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


class ResidualBlock(nn.Module):
    """Residual block with time embedding"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.SiLU(),
        )
        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block(x)
        h = h + self.time_mlp(time_emb).view(x.shape[0], -1, 1, 1)
        return h + self.residual_conv(x)


class DiffusionModel(nn.Module):
    """Simple UNet-like diffusion model"""
    
    def __init__(self, in_channels=3, model_channels=128, num_res_blocks=2, num_classes=1000):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(1, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, time_emb_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList()
        ch = model_channels
        for i in range(3):
            self.encoder.append(ResidualBlock(ch, ch * 2, time_emb_dim))
            self.encoder.append(nn.AvgPool2d(2))
            ch *= 2
        
        # Bottleneck
        self.bottleneck = ResidualBlock(ch, ch, time_emb_dim)
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(3):
            self.decoder.append(ResidualBlock(ch * 2, ch // 2, time_emb_dim))
            self.decoder.append(nn.Upsample(scale_factor=2))
            ch //= 2
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1).float() / 1000.0)
        
        # Initial convolution
        h = self.conv_in(x)
        skips = []
        
        # Encoder
        for layer in self.encoder:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
                skips.append(h)
            else:
                h = layer(h)
        
        # Bottleneck
        h = self.bottleneck(h, t_emb)
        
        # Decoder
        for layer in self.decoder:
            if isinstance(layer, ResidualBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Output
        output = self.conv_out(h)
        return output


# ==================== Training ====================

class ImageDataset(Dataset):
    """Dataset for loading images"""
    
    def __init__(self, dataset_path, image_size=32):
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.image_files = list(self.dataset_path.glob("**/*.jpg")) + \
                          list(self.dataset_path.glob("**/*.png")) + \
                          list(self.dataset_path.glob("**/*.jpeg"))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {dataset_path}")
        
        print(f"Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.image_files[idx]).convert("RGB")
        return self.transform(img)


def add_noise(x_0, t, scheduler):
    """Add noise to image according to timestep t"""
    noise = torch.randn_like(x_0)
    sqrt_alpha_cumprod = scheduler.sqrt_alphas_cumprod[t]
    sqrt_one_minus_alpha_cumprod = scheduler.sqrt_one_minus_alphas_cumprod[t]
    
    # Reshape for broadcasting
    sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1, 1)
    
    x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
    return x_t, noise


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, writer, config):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for step, x_0 in enumerate(pbar):
        x_0 = x_0.to(device)
        batch_size = x_0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
        
        # Add noise
        x_t, noise = add_noise(x_0, t, scheduler)
        
        # Predict noise
        noise_pred = model(x_t, t)
        
        # Loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if config.get("max_grad_norm"):
            nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        
        # Logging
        if step % config.get("log_interval", 100) == 0:
            writer.add_scalar("train/loss", loss.item(), epoch * len(train_loader) + step)
    
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar("train/epoch_loss", avg_loss, epoch)
    print(f"Epoch {epoch} - Avg Loss: {avg_loss:.6f}")


def validate(model, val_loader, scheduler, device, epoch, writer, config):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x_0 in val_loader:
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]
            
            t = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
            x_t, noise = add_noise(x_0, t, scheduler)
            noise_pred = model(x_t, t)
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    writer.add_scalar("val/loss", avg_loss, epoch)
    print(f"Validation Loss: {avg_loss:.6f}")
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)
    
    config = base_config.get("training", {})
    config.setdefault("batch_size", 32)
    config.setdefault("learning_rate", 1e-4)
    config.setdefault("epochs", 100)
    config.setdefault("image_size", 32)
    config.setdefault("model_channels", 128)
    config.setdefault("num_timesteps", 1000)
    config.setdefault("val_split", 0.1)
    config.setdefault("max_grad_norm", 1.0)
    config.setdefault("log_interval", 100)
    config.setdefault("save_interval", 10)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Tensorboard
    writer = SummaryWriter(output_dir / "logs")
    
    # Dataset
    dataset_path = base_config["paths"]["dataset_path"]
    print(f"Loading dataset from {dataset_path}...")
    
    dataset = ImageDataset(dataset_path, image_size=config["image_size"])
    
    # Train/val split
    val_size = int(len(dataset) * config["val_split"])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4
    )
    
    # Model
    model = DiffusionModel(
        in_channels=3,
        model_channels=config["model_channels"]
    ).to(device)
    
    # EMA
    ema = EMA(model, beta=0.9999)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"]
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )
    
    # Noise schedule
    noise_scheduler = DiffusionScheduler(num_timesteps=config["num_timesteps"])
    noise_scheduler = noise_scheduler.to(device)
    
    # Training loop
    print(f"Starting training for {config['epochs']} epochs...")
    best_val_loss = float("inf")
    
    for epoch in range(config["epochs"]):
        train_epoch(model, train_loader, optimizer, noise_scheduler, device, epoch, writer, config)
        ema.update()
        
        # Validation
        if (epoch + 1) % config.get("val_interval", 5) == 0:
            val_loss = validate(model, val_loader, noise_scheduler, device, epoch, writer, config)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        
        # Save checkpoint
        if (epoch + 1) % config["save_interval"] == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"model_epoch_{epoch}.pt")
            torch.save(ema.ema_model.state_dict(), checkpoint_dir / f"ema_model_epoch_{epoch}.pt")
        
        scheduler.step()
    
    # Save final models
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pt")
    torch.save(ema.ema_model.state_dict(), checkpoint_dir / "final_ema_model.pt")
    
    writer.close()
    print(f"Training complete! Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
