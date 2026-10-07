# pile-wm

An in-progress action-conditioned latent world model of granular pile pushing, intended to be evaluated by model-based planning in its simulator.

**Implementation paused at the DINOv2 prerequisite.** The official ViT-S/14 weight download at `dl.fbaipublicfiles.com` returned HTTP 403 from the cloud network proxy. As requested, no alternative encoder is used. Simulator and state-only data generation are implemented; the encoder/decoder, predictor, rollout evaluation, CEM planning, and combined pipeline are not yet implemented. This repository does **not** yet establish any of the four world-model criteria.

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required. From this checkout:

```bash
# Cloud machine: writable caches (home is read-only).
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.local/python
export TORCH_HOME=/workspace/.cache/torch
export MPLCONFIGDIR=/workspace/.cache/matplotlib
uv sync --frozen
uv run --frozen pytest -q
```

No accounts, credentials, or background services are needed for the completed stages. Device selection is CUDA, then MPS, then CPU; override `device` in YAML. See [DECISIONS.md](DECISIONS.md) for design choices, failures, and validation history, and [AGENTS.md](AGENTS.md) for development conventions.

## Completed stage commands

Run from the repository root. Each YAML supplies the random seed, output path, simulator parameters, and trajectory count. Data consists only of particle positions, pusher positions, and actions, with metadata containing disjoint train/validation/test trajectory IDs. Rendering is on demand. The temporally correlated random policy reflects its momentum at table boundaries.

```bash
# smoke: 32 trajectories, 40 actions each
uv run --frozen python -m scripts.random_push --config configs/smoke.yaml
uv run --frozen python -m scripts.generate_data --config configs/smoke.yaml
uv run --frozen python -m scripts.validate_data --config configs/smoke.yaml

# small: 500 trajectories; not executed in this CPU session
uv run --frozen python -m scripts.random_push --config configs/small.yaml
uv run --frozen python -m scripts.generate_data --config configs/small.yaml
uv run --frozen python -m scripts.validate_data --config configs/small.yaml

# full: 5000 trajectories; pending, run explicitly only
uv run --frozen python -m scripts.random_push --config configs/full.yaml
uv run --frozen python -m scripts.generate_data --config configs/full.yaml
uv run --frozen python -m scripts.validate_data --config configs/full.yaml
```

The generator writes to `artifacts/<config>/data/`; invoking it again regenerates that dataset. Geometry and policy measurements are saved by validation to `artifacts/<config>/data_metrics.json`. The random-pushing GIF is generated at `artifacts/<config>/random_push.gif`. Artifacts are ignored by Git.

## Measured simulator/data results

The unit suite passed **8 tests** on CPU (unit fixtures, not a trained-model experiment). The following results come from the completed `smoke` run, seed 42; raw measurements are in `artifacts/smoke/data_metrics.json`.

| Measurement | Config | Measured result |
| --- | --- | ---: |
| Trajectories / transitions | smoke | 32 / 1280 |
| Train / validation / test trajectories | smoke | 26 / 3 / 3 |
| Maximum penetration (table units) | smoke | 0.00012286 |
| Configured penetration tolerance | smoke | 0.0005 |
| Mean pusher path length (table units) | smoke | 1.239518 |
| Mean final particle displacement (table units) | smoke | 0.008132 |
| Mean cosine of adjacent actions | smoke | 0.653647 |
| Data generation wall time (seconds) | smoke | 291.57 |
| Random-pushing GIF | smoke | 40 frames, 224×224 |
| Model and planning results | smoke / small / full | Pending |

The generation wall time includes concurrent CPU work during part of the run; it is not an isolated throughput benchmark. Data generation alone took about five minutes here, so the requested end-to-end CPU smoke budget is **not yet satisfied**. Profile simulator convergence and CPU threading before completing that performance target.

Generated smoke demonstration (local artifact):

![Smoke random pushing](artifacts/smoke/random_push.gif)

## Results and remaining work

Model rollout, action-shuffle, counterfactual, mass-drift, and planning results are **unrun** for smoke, small, and full. Their requested figures, model rollout GIFs, metrics JSON, results table, and one-command pipeline are pending. The flow-matching stretch goal is also pending. Do not interpret simulator validation as evidence of a learned world model.

The environment draft includes the required `dl.fbaipublicfiles.com` domain addition and reusable dependency/start instructions. Review and save those changes in environment settings, then publish the environment. After live access is available, retry the official download and continue milestone 3:

```bash
mkdir -p "$TORCH_HOME/hub/checkpoints"
curl --fail --location \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
  --output "$TORCH_HOME/hub/checkpoints/dinov2_vits14_pretrain.pth"
```

TLS verification must remain enabled. The frozen DINOv2 encoder must be validated before implementing or claiming results for subsequent milestones. CPU end-to-end smoke runtime is not yet verified.
