from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import Conv3x3, FourierFeatures, GroupNorm, UNet


@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None


class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig, noise_conditioning: bool = True) -> None:
        super().__init__()
        self.noise_conditioning = noise_conditioning
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.act_emb = nn.Sequential(
            nn.Embedding(cfg.num_actions, cfg.cond_channels // cfg.num_steps_conditioning),
            nn.Flatten(),  # b t e -> b (t e)
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        if self.noise_conditioning:
            cond_noise = self.noise_emb(c_noise)
        else:
            cond_noise = torch.zeros_like(self.noise_emb(torch.zeros_like(c_noise)))

        # Old version
        # cond = self.cond_proj(self.noise_emb(c_noise) + self.act_emb(act))

        # New version
        cond = self.cond_proj(cond_noise + self.act_emb(act))

        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x
