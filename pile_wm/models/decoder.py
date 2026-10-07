import math
import torch
from torch import nn
import torch.nn.functional as F


class DensityDecoder(nn.Module):
    """Evaluation only. Detach at the boundary, including during optimization."""
    def __init__(self, dim=384, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, width, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width//2, 3, padding=1), nn.GELU(),
            nn.Conv2d(width//2, 2, 1))
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([math.log(math.expm1(.05)), math.log(math.expm1(1/4096))]))

    def forward(self, latent):
        if latent.shape[-2] != 256:
            raise ValueError("decoder expects a 16x16 patch grid")
        x = latent.detach().transpose(1, 2).reshape(-1, latent.shape[-1], 16, 16)
        return F.softplus(self.net(x))


def decoder_loss(prediction, truth):
    density = F.mse_loss(prediction[:, 0], truth[:, 0])
    pusher = 100*F.mse_loss(prediction[:, 1], truth[:, 1])
    mass = ((prediction.sum((-1, -2))-truth.sum((-1, -2)))/truth.sum((-1, -2)).clamp_min(1)).square().mean()
    return density+pusher+.01*mass
