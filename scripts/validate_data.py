"""Audit every saved state and report policy travel; fails on invalid geometry."""
import argparse
import json
from pathlib import Path
import torch
from pile_wm.config import load_config, setup
from pile_wm.sim import PileSim, State


def validate(config):
    setup(config)
    root = Path(config.output)/"data"
    metadata = json.loads((root/"metadata.json").read_text())
    assert metadata["config"]["trajectories"] == config.trajectories
    flat = sum(metadata["splits"].values(), [])
    assert sorted(flat) == list(range(config.trajectories))
    sim = PileSim(config.sim)
    maximum = 0.
    travel, particle_motion, correlations = [], [], []
    for i in range(config.trajectories):
        item = torch.load(root/f"trajectory_{i:05d}.pt", weights_only=True)
        x, p, a = item["particles"], item["pusher"], item["actions"]
        assert x.shape == (config.steps+1, config.sim.particles, 2)
        assert p.shape == (config.steps+1, 2) and a.shape == (config.steps, 2)
        assert all(torch.isfinite(t).all() for t in (x, p, a))
        assert a.abs().max() <= 1
        assert x.min() >= config.sim.particle_radius and x.max() <= 1-config.sim.particle_radius
        assert p.min() >= config.sim.pusher_radius and p.max() <= 1-config.sim.pusher_radius
        for offset in range(0, len(x), 8):
            maximum = max(maximum, float(sim.penetration(State(x[offset:offset+8], p[offset:offset+8]))))
        travel.append(float((p[1:]-p[:-1]).norm(dim=-1).sum()))
        particle_motion.append(float((x[-1]-x[0]).norm(dim=-1).mean()))
        correlations.append(float(torch.nn.functional.cosine_similarity(a[1:], a[:-1], dim=-1).mean()))
    metrics = {"config": config.name, "trajectories": config.trajectories,
               "transitions": config.trajectories*config.steps,
               "split_counts": {k: len(v) for k, v in metadata["splits"].items()},
               "generation_seconds": metadata["generation_seconds"],
               "max_penetration": maximum,
               "penetration_tolerance": config.sim.tolerance,
               "mean_pusher_path_length": sum(travel)/len(travel),
               "mean_particle_displacement": sum(particle_motion)/len(particle_motion),
               "mean_adjacent_action_cosine": sum(correlations)/len(correlations),
               "geometry_passed": maximum <= config.sim.tolerance}
    (Path(config.output)/"data_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    assert metrics["geometry_passed"], "contact penetration exceeds configured tolerance"
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    validate(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
