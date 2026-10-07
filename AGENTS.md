# pile-wm

Use the existing checkout; cloud tasks are isolated. Do not create worktrees unless explicitly requested.

Python 3.11, uv, PyTorch, dataclasses/YAML. Source: `pile_wm/`; simulator: `pile_wm/sim/`; stage entry points: `scripts/`; reproducible presets: `configs/`; tests: `tests/`. Generated datasets, checkpoints, figures and metrics belong in ignored `artifacts/<config>/`.

In this cloud environment export `UV_CACHE_DIR=/workspace/.cache/uv`, `UV_PYTHON_INSTALL_DIR=/workspace/.local/python`, and `TORCH_HOME=/workspace/.cache/torch` before using uv. Install with `uv sync --frozen`; test with `uv run pytest`. Generate simulator demo with `uv run python -m scripts.random_push --config configs/smoke.yaml`.

Seed all experiments from configuration; auto-select CUDA/MPS/CPU. Keep predictor and decoder optimization separate. Never replace DINOv2 in experiments; test doubles are for unit tests only. Record design choices and failures in DECISIONS.md. Commit each completed, tested milestone. Document only actually measured results, explicitly label their config, and leave full results pending. Never run full automatically.
