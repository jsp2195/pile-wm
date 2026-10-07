"""Official DINOv2 ViT-S/14 weights in the equivalent timm architecture."""
import hashlib
import os
from pathlib import Path
import urllib.request

import torch
from torch import nn
import torch.nn.functional as F

URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
# Recorded from the official checkpoint retrieved via verified HTTPS.
SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def checkpoint_path():
    root = Path(os.environ.get("TORCH_HOME", "/workspace/.cache/torch"))/"hub/checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    path = root/"dinov2_vits14_pretrain.pth"
    if not path.exists():
        temporary = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(URL, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
                raise RuntimeError("DINOv2 checkpoint checksum mismatch")
            temporary.replace(path)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Official DINOv2 weights unavailable; stop, do not substitute an encoder") from exc
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("DINOv2 checkpoint checksum mismatch; remove corrupt cache and retry")
    return path


class DinoEncoder(nn.Module):
    dim = 384
    tokens = 256

    def __init__(self):
        super().__init__()
        import timm
        from timm.models.vision_transformer import checkpoint_filter_fn
        self.backbone = timm.create_model("vit_small_patch14_dinov2", pretrained=False, img_size=224, num_classes=0)
        weights = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)
        # Match original DINOv2 interpolate_pos_encoding: bicubic, offset=0.1,
        # antialias=False, at 224px. CLS is kept internally, excluded from output.
        pos = weights["pos_embed"]
        side = int((pos.shape[1]-1)**0.5)
        patches = pos[:, 1:].reshape(1, side, side, 384).permute(0, 3, 1, 2)
        patches = F.interpolate(patches.float(), scale_factor=(16.1/side, 16.1/side), mode="bicubic", align_corners=False, antialias=False)
        weights["pos_embed"] = torch.cat((pos[:, :1], patches.permute(0, 2, 3, 1).reshape(1, 256, 384)), 1)
        self.backbone.load_state_dict(checkpoint_filter_fn(weights, self.backbone), strict=True)
        self.backbone.requires_grad_(False).eval()
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def train(self, mode=True):
        # Frozen means eval mode even when a containing module is trained.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        if images.shape[1:] != (3, 224, 224):
            raise ValueError("DINOv2 requires [batch,3,224,224] images")
        return self.backbone.forward_features((images-self.image_mean)/self.image_std)[:, 1:]


class Standardizer(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, batches):
        count, total, square = 0, None, None
        for batch in batches:
            # Accumulate in CPU float64: MPS does not support float64 tensors.
            x = batch.detach().reshape(-1, batch.shape[-1]).cpu().double()
            count += len(x)
            total = x.sum(0) if total is None else total+x.sum(0)
            square = x.square().sum(0) if square is None else square+x.square().sum(0)
        if count == 0:
            raise ValueError("normalization needs training frames")
        mean = total/count
        std = (square/count-mean.square()).clamp_min(1e-8).sqrt()
        self.mean.copy_(mean.float())
        self.std.copy_(std.float())
        return count

    def forward(self, x):
        return (x-self.mean)/self.std
