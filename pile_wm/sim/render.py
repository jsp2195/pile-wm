"""Bounded-memory batched rasterization: pixel tiles, never B*N*H*W."""
import torch


def _discs(points, radius, size, chunk=16):
    b = points.shape[0]
    grid = (torch.arange(size, device=points.device, dtype=points.dtype)+0.5)/size
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    pixels = torch.stack((xx, yy), -1).reshape(-1, 2)
    output = torch.zeros(b, size*size, device=points.device)
    # Bound intermediates to B*chunk*H*W distances, independent of N.
    for start in range(0, points.shape[1], chunk):
        d = (points[:, start:start+chunk, None]-pixels[None, None]).square().sum(-1)
        output = torch.maximum(output, (d <= radius**2).any(1).float())
    return output.reshape(b, size, size)


@torch.no_grad()
def render(state, config, size=224):
    particles = _discs(state.particles, config.particle_radius, size)
    pusher = _discs(state.pusher[:, None], config.pusher_radius, size)
    bg = torch.tensor([0.09, 0.12, 0.16], device=particles.device)[None, :, None, None]
    pc = torch.tensor([0.90, 0.70, 0.25], device=particles.device)[None, :, None, None]
    uc = torch.tensor([0.20, 0.70, 0.90], device=particles.device)[None, :, None, None]
    image = bg*(1-particles[:, None]) + pc*particles[:, None]
    return image*(1-pusher[:, None]) + uc*pusher[:, None]


@torch.no_grad()
def state_maps(state, size=64):
    """Bilinear particle count density (sum=N), and unit-mass pusher heatmap."""
    def splat(points):
        xy = (points*size-0.5).clamp(0, size-1)
        low = xy.floor().long()
        frac = xy-low
        result = torch.zeros(points.shape[0], size*size, device=points.device)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            index = (low[..., 1]+dy).clamp_max(size-1)*size+(low[..., 0]+dx).clamp_max(size-1)
            weights = (frac[..., 0] if dx else 1-frac[..., 0])*(frac[..., 1] if dy else 1-frac[..., 1])
            result.scatter_add_(1, index, weights)
        return result.reshape(-1, size, size)
    return torch.stack((splat(state.particles), splat(state.pusher[:, None])), 1)
