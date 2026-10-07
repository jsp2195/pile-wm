from dataclasses import asdict, dataclass, field
from pathlib import Path
import random

import numpy as np
import torch
import yaml


@dataclass
class SimConfig:
    particles: int = 200
    particle_radius: float = 0.012
    pusher_radius: float = 0.04
    max_displacement: float = 0.04
    substeps: int = 5
    contact_iterations: int = 40
    init_iterations: int = 300
    tolerance: float = 0.0005


@dataclass
class Config:
    name: str = "smoke"
    seed: int = 42
    device: str = "auto"
    threads: int = 4
    output: str = "artifacts/smoke"
    trajectories: int = 32
    steps: int = 40
    data_batch_size: int = 8
    sim: SimConfig = field(default_factory=SimConfig)

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))


def load_config(path):
    values = yaml.safe_load(Path(path).read_text())
    values["sim"] = SimConfig(**values.get("sim", {}))
    return Config(**values)


def setup(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    if config.device != "auto":
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
