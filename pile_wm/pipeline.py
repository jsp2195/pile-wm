"""Single-command reproducible pipeline; no implicit full run or metric tuning."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from pile_wm.config import setup
from pile_wm.data import generate
from pile_wm.sim.environment import SIMULATOR_VERSION
from pile_wm.training import train_decoder, train_model, save_json
from pile_wm.evaluation import evaluate
from pile_wm.planning import run_planning


def data_matches(config):
    path = Path(config.output)/"data/metadata.json"
    if not path.exists():
        return False
    metadata = json.loads(path.read_text())
    if metadata.get("simulator_version") != SIMULATOR_VERSION:
        return False
    current = asdict(config)
    fields = ("seed", "trajectories", "steps", "data_batch_size", "sim")
    return all(metadata["config"].get(key) == current[key] for key in fields) and all(
        (path.parent/f"trajectory_{i:05d}.pt").exists() for i in range(config.trajectories))


def run_pipeline(config, regenerate=False):
    from scripts.validate_data import validate
    from pile_wm.reporting import create_report
    from scripts.random_push import make_gif
    root = Path(config.output)
    root.mkdir(parents=True, exist_ok=True)
    previous_runs = []
    if (root/"metrics.json").exists():
        old = json.loads((root/"metrics.json").read_text())
        if old.get("resolved_config") == asdict(config) and "total_seconds" in old:
            previous_runs = old.get("previous_runs", []) + [{
                key: old[key] for key in ("timestamp_utc", "total_seconds", "dataset_reused", "tests_passed")
            }]
    config.save(root/"resolved_config.yaml")
    started = time.perf_counter()
    checks = subprocess.run([sys.executable, "-m", "pytest", "-q", "--color=no"], capture_output=True, text=True)
    (root/"tests.log").write_text(checks.stdout+checks.stderr)
    print(checks.stdout, flush=True)
    checks.check_returncode()
    match = re.search(r"(\d+) passed", checks.stdout)
    if not match:
        raise RuntimeError("No passing tests reported by pytest")
    stages = {}
    reused = not regenerate and data_matches(config)
    for name, operation in (
        ("data", lambda: None if reused else generate(config, setup(config))),
        ("data_validation", lambda: validate(config)),
        ("random_gif", lambda: make_gif(config)),
        ("decoder", lambda: train_decoder(config, setup(config))),
        ("predictor", lambda: train_model(config, setup(config))),
        ("evaluation", lambda: evaluate(config, setup(config))),
        ("planning", lambda: run_planning(config, setup(config))),
    ):
        before = time.perf_counter()
        print(f"Stage: {name}", flush=True)
        operation()
        stages[name] = time.perf_counter()-before
    metrics = {"config": config.name, "resolved_config": asdict(config),
               "previous_runs": previous_runs,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "tests_passed": int(match.group(1)), "dataset_reused": reused,
               "stages_seconds": stages,
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for folder in ("pile_wm", "scripts", "configs") for p in sorted(Path(folder).rglob("*")) if p.is_file() and p.suffix in (".py", ".yaml")}}
    for name, filename in (("data", "data_metrics"), ("decoder", "decoder_metrics"), ("training", "training_metrics"),
                           ("evaluation", "evaluation_metrics"), ("planning", "planning_metrics")):
        metrics[name] = json.loads((root/(filename+".json")).read_text())
    metrics["world_model_criteria"] = {
        "next_state_beats_copy": metrics["evaluation"]["beats_copy_mean_mse"],
        "autoregressive_horizons_executed": metrics["evaluation"]["horizons"],
        "action_shuffle_passes": metrics["evaluation"]["action_shuffle"]["passes"],
        "counterfactual_density_correlation": metrics["evaluation"]["counterfactual"]["density_correlation"],
        "planning_advantage_demonstrated": metrics["planning"]["planning_advantage_demonstrated"]}
    density_correlation = metrics["world_model_criteria"]["counterfactual_density_correlation"]
    metrics["world_model_demonstrated"] = all((metrics["world_model_criteria"]["next_state_beats_copy"],
                                               metrics["world_model_criteria"]["action_shuffle_passes"],
                                               density_correlation is not None and density_correlation > 0,
                                               metrics["world_model_criteria"]["planning_advantage_demonstrated"]))
    metrics["seconds_before_reporting"] = time.perf_counter()-started
    save_json(root/"metrics.json", metrics)
    report_start = time.perf_counter()
    create_report(config, metrics)
    metrics["stages_seconds"]["reporting"] = time.perf_counter()-report_start
    metrics["total_seconds"] = time.perf_counter()-started
    save_json(root/"metrics.json", metrics)
    # Refresh report text/metrics with final wall time, without redrawing figures.
    create_report(config, metrics, figures=False)
    print(f"Pipeline complete: {metrics['total_seconds']:.2f}s; world-model competence demonstrated: {metrics['world_model_demonstrated']}", flush=True)
    return metrics
