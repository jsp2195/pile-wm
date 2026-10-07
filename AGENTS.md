# pile-wm

Use the existing checkout; cloud tasks are isolated. Do not create worktrees unless explicitly requested.

Python 3.11, uv, PyTorch, dataclasses/YAML. Source: `pile_wm/`; simulator: `pile_wm/sim/`; stage entry points: `scripts/`; reproducible presets: `configs/`; tests: `tests/`. Generated datasets, checkpoints, figures and metrics belong in ignored `artifacts/<config>/`.

In this cloud environment export `UV_CACHE_DIR=/workspace/.cache/uv`, `UV_PYTHON_INSTALL_DIR=/workspace/.local/python`, and `TORCH_HOME=/workspace/.cache/torch` before using uv. Install with `uv sync --frozen`; test with `uv run pytest`. Generate simulator demo with `uv run python -m scripts.random_push --config configs/smoke.yaml`.

Generate state-only trajectories with `uv run python -m scripts.generate_data --config configs/smoke.yaml`; audit every state with `uv run python -m scripts.validate_data --config configs/smoke.yaml`. The YAML presets define smoke=32, small=500, full=5000 trajectories of 40 actions. Generation replaces files in the selected output directory. Data splits are by trajectory; never compute normalization statistics from validation/test data.

Current blocker: the DINOv2 checkpoint host returned proxy HTTP 403. A domain addition is saved in environment settings but live access has not been established. Milestones 3 onward are pending; retry the official download after network settings are applied. Stop if weights remain unavailable; no substitute encoder is allowed. The current simulator/data commands are not an end-to-end world-model pipeline.

Seed all experiments from configuration; auto-select CUDA/MPS/CPU. Keep predictor and decoder optimization separate. Never replace DINOv2 in experiments; test doubles are for unit tests only. Record design choices and failures in DECISIONS.md. Commit each completed, tested milestone. Document only actually measured results, explicitly label their config, and leave full results pending. Never run full automatically.
