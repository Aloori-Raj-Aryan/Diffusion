import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embedder import SinusoidalPositionEmbedding
from model.basicblocks import EncodingBlocks, DecodingBlocks, ResBlock


class UNet(nn.Module):
    """Basic UNet architecture for diffusion models with time conditioning."""

    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg["training"]["image_size"]
        self.time_embedding_dim = cfg["training"]["time_embedding_dim"]
        self.channels = cfg["training"]["model_channels"]

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(self.time_embedding_dim),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )

        # ── Encoder Level 1 (ch=128) ──────────────────────────────────────────
        self.encoding_blocks = nn.ModuleList()
        self.decoding_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        for i in range(1, len(self.channels)):
            self.encoding_blocks.append(EncodingBlocks(self.channels[i-1], self.channels[i], self.time_embedding_dim))
        
        for i in range(len(self.channels)-1, 0, -1):
            self.decoding_blocks.append(DecodingBlocks(self.channels[i]*2, self.channels[i-1], self.time_embedding_dim))

        for _ in range(2):
            self.res_blocks.append(ResBlock(self.channels[-1], self.channels[-1], self.time_embedding_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        skips = []
        for module in self.encoding_blocks:
            skip, x = module(x, t_emb)
            skips.append(skip)

        for module in self.res_blocks:
            x = module(x, t_emb)
        
        for module in self.res_blocks:
             x = module(x, t_emb)

        for i, module in enumerate(self.decoding_blocks):
            x = module(x, skips[-(i+1)], t_emb)
        
        return x
    
if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    model = UNet(cfg)
    input_ = torch.randn(8, 3, cfg["training"]["image_size"], cfg["training"]["image_size"])
    t = torch.randint(0, cfg["training"]["num_timesteps"], (8,))
    output = model(input_, t)
    print(output.shape)