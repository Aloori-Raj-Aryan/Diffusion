import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embedder import SinusoidalPositionEmbedding
from model.basicblocks import ResBlock, Downsample, Upsample


class UNet(nn.Module):
    """Basic UNet architecture for diffusion models with time conditioning."""

    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        time_embedding_dim: int = 512,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.time_embedding_dim = time_embedding_dim

        # Hardcoded hyperparameters

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(128),
            nn.Linear(128, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        # ── Encoder Level 1 (ch=128) ──────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.enc1_block1 = ResBlock(128, 128, time_embedding_dim)
        self.enc1_block2 = ResBlock(128, 128, time_embedding_dim)
        self.down1 = Downsample(128)

        # ── Encoder Level 2 (ch=256) ──────────────────────────────────────────
        self.enc2_block1 = ResBlock(128, 256, time_embedding_dim)
        self.enc2_block2 = ResBlock(256, 256, time_embedding_dim)
        self.down2 = Downsample(256)

        # ── Encoder Level 3 (ch=512) ──────────────────────────────────────────
        self.enc3_block1 = ResBlock(256, 512, time_embedding_dim)
        self.enc3_block2 = ResBlock(512, 512, time_embedding_dim)

        # ── Bottleneck (ch=512) ────────────────────────────────────────────────
        self.mid_block1 = ResBlock(512, 512, time_embedding_dim)
        self.mid_block2 = ResBlock(512, 512, time_embedding_dim)

        # ── Decoder Level 3 (ch=512) ──────────────────────────────────────────
        self.dec3_block1 = ResBlock(512 + 512, 512, time_embedding_dim)
        self.dec3_block2 = ResBlock(512, 512, time_embedding_dim)
        self.up3 = Upsample(512)

        # ── Decoder Level 2 (ch=256) ──────────────────────────────────────────
        self.dec2_block1 = ResBlock(512 + 256, 256, time_embedding_dim)
        self.dec2_block2 = ResBlock(256, 256, time_embedding_dim)
        self.up2 = Upsample(256)

        # ── Decoder Level 1 (ch=128) ──────────────────────────────────────────
        self.dec1_block1 = ResBlock(256 + 128, 128, time_embedding_dim)
        self.dec1_block2 = ResBlock(128, 128, time_embedding_dim)

        # ── Output ─────────────────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(8, 128)
        self.out_conv = nn.Conv2d(128, 3, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            t: Time tensor of shape (batch_size,)

        Returns:
            Output tensor of shape (batch_size, 3, height, width)
        """
        t_emb = self.time_embed(t)

        # ── Encoder ────────────────────────────────────────────────────────────
        h = self.input_conv(x)
        
        # Level 1 (32x32)
        h1 = self.enc1_block1(h, t_emb)
        h1 = self.enc1_block2(h1, t_emb)
        h = self.down1(h1)

        # Level 2 (16x16)
        h2 = self.enc2_block1(h, t_emb)
        h2 = self.enc2_block2(h2, t_emb)
        h = self.down2(h2)

        # Level 3 (8x8)
        h3 = self.enc3_block1(h, t_emb)
        h3 = self.enc3_block2(h3, t_emb)

        # ── Bottleneck (8x8) ───────────────────────────────────────────────────
        h = self.mid_block1(h3, t_emb)
        h = self.mid_block2(h, t_emb)

        # ── Decoder ────────────────────────────────────────────────────────────
        # Level 3 (8x8)
        h = torch.cat([h, h3], dim=1)
        h = self.dec3_block1(h, t_emb)
        h = self.dec3_block2(h, t_emb)
        h = self.up3(h)

        # Level 2 (16x16)
        h = torch.cat([h, h2], dim=1)
        h = self.dec2_block1(h, t_emb)
        h = self.dec2_block2(h, t_emb)
        h = self.up2(h)

        # Level 1 (32x32)
        h = torch.cat([h, h1], dim=1)
        h = self.dec1_block1(h, t_emb)
        h = self.dec1_block2(h, t_emb)

        # ── Output ─────────────────────────────────────────────────────────────
        return self.out_conv(F.silu(self.out_norm(h)))
