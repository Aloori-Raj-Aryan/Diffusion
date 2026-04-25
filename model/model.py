import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embedder import SinusoidalPositionEmbedding
from model.basicblocks import ResBlock, AttentionBlock, Downsample, Upsample

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
