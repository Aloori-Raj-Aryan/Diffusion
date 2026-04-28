import os
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from PIL import Image


# ==================== Import from train.py ====================

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
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
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
            nn.BatchNorm2d(model_channels),
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


# ==================== Inference ====================

class DiffusionSampler:
    """DDPM sampling process"""
    
    def __init__(self, model, scheduler, device, guidance_scale=1.0):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.guidance_scale = guidance_scale
    
    @torch.no_grad()
    def sample(self, batch_size, image_size=32, num_inference_steps=None):
        """Sample from the diffusion model"""
        
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_timesteps
        
        # Start from pure noise
        x_t = torch.randn(batch_size, 3, image_size, image_size, device=self.device)
        
        # Reverse diffusion process
        step_size = self.scheduler.num_timesteps // num_inference_steps
        timesteps = list(range(self.scheduler.num_timesteps - 1, 0, -step_size))
        
        for t_idx in tqdm(timesteps, desc="Sampling"):
            t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            
            # Predict noise
            pred_noise = self.model(x_t, t)
            
            # Compute mean
            alpha_t = self.scheduler.alphas[t_idx]
            alpha_cumprod_t = self.scheduler.alphas_cumprod[t_idx]
            alpha_cumprod_t_prev = self.scheduler.alphas_cumprod_prev[t_idx]
            
            posterior_mean_coeff1 = (
                torch.sqrt(alpha_cumprod_t_prev) * self.scheduler.betas[t_idx] /
                (1.0 - alpha_cumprod_t)
            )
            posterior_mean_coeff2 = (
                torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_t_prev) /
                (1.0 - alpha_cumprod_t)
            )
            
            mean = posterior_mean_coeff1.view(-1, 1, 1, 1) * x_t + \
                   posterior_mean_coeff2.view(-1, 1, 1, 1) * pred_noise
            
            # Add variance
            variance = self.scheduler.posterior_variance[t_idx]
            if t_idx > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(variance).view(-1, 1, 1, 1) * noise
            else:
                x_t = mean
        
        # Denormalize
        x_t = (x_t + 1) / 2
        x_t = torch.clamp(x_t, 0, 1)
        
        return x_t


def load_model(checkpoint_path, device, model_channels=128):
    """Load trained diffusion model"""
    model = DiffusionModel(in_channels=3, model_channels=model_channels).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def save_samples(samples, output_dir, prefix="sample"):
    """Save generated samples"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save grid
    from torchvision.utils import make_grid
    grid = make_grid(samples, nrow=4, normalize=True)
    save_image(grid, output_dir / f"{prefix}_grid.png")
    
    # Save individual samples
    for i, sample in enumerate(samples):
        save_image(sample, output_dir / f"{prefix}_{i:04d}.png")
    
    print(f"Saved samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=100, help="Number of diffusion steps")
    parser.add_argument("--output_dir", type=str, default="generated_samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)
    
    config = base_config.get("training", {})
    image_size = config.get("image_size", 32)
    model_channels = config.get("model_channels", 128)
    num_timesteps = config.get("num_timesteps", 1000)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Loading model from {args.checkpoint}...")
    
    # Load model
    model = load_model(args.checkpoint, device, model_channels)
    
    # Create scheduler
    scheduler = DiffusionScheduler(num_timesteps=num_timesteps)
    scheduler = scheduler.to(device)
    
    # Create sampler
    sampler = DiffusionSampler(model, scheduler, device)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    samples = sampler.sample(
        batch_size=args.num_samples,
        image_size=image_size,
        num_inference_steps=args.num_steps
    )
    
    # Save samples
    save_samples(samples, args.output_dir)


if __name__ == "__main__":
    main()
