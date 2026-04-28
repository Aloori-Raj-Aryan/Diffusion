import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch * 2))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)

        x_copy = x
        x = self.norm1(x)
        x = self.conv1(F.silu(x))
        x = self.norm2(x)
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        x = self.conv2(F.silu(x))
        return x + self.skip(x_copy)

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
    
class EncodingBlocks(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_emb_dim)
        self.down = Downsample(out_ch)
    
    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x, self.down(x)

class DecodingBlocks(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = Upsample(in_ch//2)
        self.res1 = ResBlock(in_ch, out_ch, time_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_emb_dim)
    
    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


