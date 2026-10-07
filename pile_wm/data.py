"""State-only trajectories, trajectory-disjoint splits, and on-demand rendering."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
from torch.utils.data import Dataset
from pile_wm.sim import PileSim, State, render
from pile_wm.sim.environment import SIMULATOR_VERSION


def split_ids(count, seed):
    if count < 5:
        raise ValueError("need at least five trajectories for nonempty splits")
    ids = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, round(0.1*count))
    n_test = max(1, round(0.1*count))
    return {"train": ids[:count-n_val-n_test].tolist(),
            "val": ids[count-n_val-n_test:count-n_test].tolist(),
            "test": ids[count-n_test:].tolist()}


@torch.no_grad()
def generate(config, device, destination=None):
    root = Path(destination or Path(config.output)/"data")
    root.mkdir(parents=True, exist_ok=True)
    # An interrupted regeneration must not expose a mixed dataset as complete.
    (root/"metadata.json").unlink(missing_ok=True)
    sim = PileSim(config.sim, device)
    g = torch.Generator(device="cpu").manual_seed(config.seed+101)
    started = time.perf_counter()
    for first in range(0, config.trajectories, config.data_batch_size):
        b = min(config.data_batch_size, config.trajectories-first)
        # Per-trajectory reset seeds keep initial states independent of batching.
        states = [sim.reset(1, config.seed+i) for i in range(first, first+b)]
        state = State(torch.cat([s.particles for s in states]), torch.cat([s.pusher for s in states]))
        xs, ps, actions = [state.particles.cpu()], [state.pusher.cpu()], []
        a = (torch.rand(b, 2, generator=g)*2-1).to(device)
        for _ in range(config.steps):
            a = 0.85*a + 0.45*torch.randn(b, 2, generator=g).to(device)
            # Reflect correlated policy momentum at walls instead of pinning there.
            a = torch.where(state.pusher < 0.08, a.abs(), a)
            a = torch.where(state.pusher > 0.92, -a.abs(), a).clamp(-1, 1)
            actions.append(a.cpu())
            state = sim.step(state, a)
            xs.append(state.particles.cpu())
            ps.append(state.pusher.cpu())
        xs, ps, actions = torch.stack(xs, 1), torch.stack(ps, 1), torch.stack(actions, 1)
        for offset in range(b):
            torch.save({"particles": xs[offset], "pusher": ps[offset], "actions": actions[offset]}, root/f"trajectory_{first+offset:05d}.pt")
        print(f"Generated {first+b}/{config.trajectories} trajectories", flush=True)
    metadata = {"config": asdict(config), "splits": split_ids(config.trajectories, config.seed),
                "generation_seconds": time.perf_counter()-started,
                "simulator_version": SIMULATOR_VERSION,
                "schema": "particles[T+1,N,2], pusher[T+1,2], actions[T,2]"}
    (root/"metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


class Trajectories(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.metadata = json.loads((self.root/"metadata.json").read_text())
        self.ids = self.metadata["splits"][split]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return torch.load(self.root/f"trajectory_{self.ids[index]:05d}.pt", weights_only=True)

    @staticmethod
    def images(trajectory, config, device="cpu"):
        # Caller controls the time/batch slice so 224px frames need not reside on disk.
        return render(State(trajectory["particles"].to(device), trajectory["pusher"].to(device)), config)
