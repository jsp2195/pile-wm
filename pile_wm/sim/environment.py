"""Batched overdamped position-based disc contacts; no velocities or inertia."""
from dataclasses import dataclass
import math
import torch
from pile_wm.config import SimConfig

SIMULATOR_VERSION = 2


@dataclass
class State:
    particles: torch.Tensor  # [B,N,2], normalized table coordinates
    pusher: torch.Tensor  # [B,2]

    def clone(self):
        return State(self.particles.clone(), self.pusher.clone())

    def to(self, device):
        return State(self.particles.to(device), self.pusher.to(device))


class PileSim:
    def __init__(self, config=None, device="cpu"):
        self.config = config or SimConfig()
        self.device = torch.device(device)
        n = self.config.particles
        self.eye = torch.eye(n, dtype=torch.bool, device=self.device)[None]
        index = torch.arange(n, device=self.device)
        sign = torch.sign(index[:, None]-index[None, :]).float()
        self.tie_direction = torch.stack((sign, torch.zeros_like(sign)), -1)[None]

    @torch.no_grad()
    def reset(self, batch_size=1, seed=0):
        """A seeded mixture of scatter and 1--3 Gaussian clusters, then settle."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        c = self.config
        particles = []
        for _ in range(batch_size):
            k = int(torch.randint(1, 4, (), generator=g))
            centers = 0.2 + 0.6 * torch.rand(k, 2, generator=g)
            labels = torch.randint(k, (c.particles,), generator=g)
            cluster = centers[labels] + 0.11 * torch.randn(c.particles, 2, generator=g)
            uniform = torch.rand(c.particles, 2, generator=g)
            mask = torch.rand(c.particles, 1, generator=g) < 0.2
            particles.append(torch.where(mask, uniform, cluster))
        x = torch.stack(particles).to(self.device).clamp(c.particle_radius, 1-c.particle_radius)
        p = (c.pusher_radius + (1-2*c.pusher_radius)*torch.rand(batch_size, 2, generator=g)).to(self.device)
        return State(self._settle(self._resolve(x, p, c.init_iterations), p), p)

    def _penetration_by_env(self, x, p):
        c = self.config
        dist = (x[:, :, None]-x[:, None, :]).norm(dim=-1)
        dist.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))
        pp = (2*c.particle_radius-dist).clamp_min(0).amax((1, 2))
        push = (c.particle_radius+c.pusher_radius-(x-p[:, None]).norm(dim=-1)).clamp_min(0).amax(1)
        return torch.maximum(pp, push)

    def _settle(self, x, p):
        # Fixed iteration budgets alone fail on dense contacts. Freeze converged
        # environments independently so results do not depend on batch neighbors.
        for _ in range(20):
            active = self._penetration_by_env(x, p) > self.config.tolerance/4
            if not active.any():
                return x
            resolved = self._resolve(x, p, self.config.contact_iterations)
            x = torch.where(active[:, None, None], resolved, x)
        if (self._penetration_by_env(x, p) > self.config.tolerance).any():
            raise RuntimeError("Contact solver did not converge; increase contact_iterations")
        return x

    def _resolve(self, x, p, iterations):
        c = self.config
        n = x.shape[1]
        # Dense Jacobi contacts: simultaneous updates preserve batching equivalence.
        active = torch.ones(len(x), dtype=torch.bool, device=x.device)
        for iteration in range(iterations):
            previous = x
            d = x - p[:, None]
            dist = d.norm(dim=-1, keepdim=True)
            fallback = torch.zeros_like(d)
            fallback[..., 0] = 1
            direction = torch.where(dist > 1e-9, d / dist.clamp_min(1e-9), fallback)
            x = x + direction * (c.particle_radius+c.pusher_radius-dist).clamp_min(0)
            d = x[:, :, None] - x[:, None, :]
            dist = d.norm(dim=-1, keepdim=True)
            overlap = (2*c.particle_radius-dist).clamp_min(0).masked_fill(self.eye[..., None], 0)
            # Stable antisymmetric tie-breaking for coincident particles.
            direction = torch.where(dist > 1e-9, d/dist.clamp_min(1e-9), self.tie_direction)
            contacts = (overlap > 0).sum(2).clamp_min(2)
            correction = (direction*overlap).sum(2) / contacts
            x = (x + correction).clamp(c.particle_radius, 1-c.particle_radius)
            x = torch.where(active[:, None, None], x, previous)
            if (iteration+1) % 4 == 0:
                active = active & (self._penetration_by_env(x, p) > c.tolerance/8)
                if not active.any():
                    break
        return x

    @torch.no_grad()
    def step(self, state, action):
        c = self.config
        if action.shape != state.pusher.shape:
            raise ValueError("action must have shape [batch,2]")
        action = action.to(self.device).clamp(-1, 1)
        delta = action*c.max_displacement
        # Norm <= max displacement, even for diagonal actions.
        delta = delta / action.norm(dim=-1, keepdim=True).clamp_min(1)
        count = max(c.substeps, math.ceil(c.max_displacement/(0.8*c.particle_radius)))
        p0 = state.pusher
        target = (p0+delta).clamp(c.pusher_radius, 1-c.pusher_radius)
        x = state.particles.clone()
        for i in range(count):
            p = p0 + (target-p0)*((i+1)/count)
            x = self._resolve(x, p, c.contact_iterations)
        return State(self._settle(x, target), target)

    def penetration(self, state):
        return self._penetration_by_env(state.particles, state.pusher).amax()
