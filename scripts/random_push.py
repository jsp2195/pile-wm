"""Run with: uv run python -m scripts.random_push --config configs/smoke.yaml"""
import argparse
from pathlib import Path
import torch
from PIL import Image
from pile_wm.config import load_config, setup
from pile_wm.sim import PileSim, render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = setup(cfg)
    sim = PileSim(cfg.sim, device)
    state = sim.reset(seed=cfg.seed)
    frames = []
    action = torch.zeros(1, 2, device=device)
    for _ in range(cfg.steps):
        frames.append(Image.fromarray((render(state, cfg.sim)[0].permute(1, 2, 0).cpu().numpy()*255).astype("uint8")))
        action = 0.85*action + 0.5*torch.randn_like(action)
        state = sim.step(state, action.clamp(-1, 1))
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    frames[0].save(output/"random_push.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(output/"random_push.gif")


if __name__ == "__main__":
    main()
