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
class TrainConfig:
    statistics_frames: int = 32
    decoder_steps: int = 20
    model_steps: int = 4
    batch_size: int = 1
    encoder_batch_size: int = 8
    learning_rate: float = 0.0001
    decoder_learning_rate: float = 0.001
    multistep_weight: float = 1.0


@dataclass
class ModelConfig:
    dim: int = 384
    depth: int = 6
    heads: int = 6
    context: int = 3


@dataclass
class EvalConfig:
    trajectories: int = 3
    horizon: int = 30
    counterfactual_k: int = 3
    counterfactual_horizon: int = 5
    occupancy_threshold: float = 0.1
    shuffle_gap_threshold: float = 0.05


@dataclass
class PlanConfig:
    episodes: int = 2
    steps: int = 3
    horizon: int = 2
    population: int = 4
    elites: int = 2
    iterations: int = 1
    chunk_size: int = 4


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
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    planning: PlanConfig = field(default_factory=PlanConfig)

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))


def load_config(path):
    values = yaml.safe_load(Path(path).read_text())
    for name, cls in (("sim", SimConfig), ("train", TrainConfig), ("model", ModelConfig),
                      ("evaluation", EvalConfig), ("planning", PlanConfig)):
        values[name] = cls(**values.get(name, {}))
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
