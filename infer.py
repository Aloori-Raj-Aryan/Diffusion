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

from utils.train_pipeline import DiffusionLightning
from utils.scheduler import DDPMScheduler
from utils.sampler import ddpm_sample


def load_model(checkpoint_path, device, cfg):
    model = DiffusionLightning.load_from_checkpoint(checkpoint_path, cfg=cfg)
    model.eval()
    model.to(device)
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
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Loading model from {args.checkpoint}...")
    
    # Load model
    model = load_model(args.checkpoint, device, base_config)
    
    # Create scheduler
    scheduler = DDPMScheduler(num_timesteps=base_config["inference"]["num_inference_steps"])
    scheduler = scheduler.to(device)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    samples = ddpm_sample(
        model=model,
        scheduler=scheduler,
        shape=(args.num_samples, 3, base_config["training"]["image_size"], base_config["training"]["image_size"]),
        device=device,
        num_steps=args.num_steps
    )
    
    # Save samples
    save_samples(samples, args.output_dir)


if __name__ == "__main__":
    main()
