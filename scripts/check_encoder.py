"""Real-weight integration check, separate from the offline unit suite."""
import json
from pathlib import Path
import time
import torch
from pile_wm.config import Config, setup
from pile_wm.models.encoder import DinoEncoder, SHA256
from pile_wm.sim import PileSim, render


def main():
    cfg = Config()
    device = setup(cfg)
    encoder = DinoEncoder().to(device)
    sim = PileSim(cfg.sim, device)
    images = render(sim.reset(1, cfg.seed), cfg.sim)
    start = time.perf_counter()
    z = encoder(images)
    elapsed = time.perf_counter()-start
    assert z.shape == (1, 256, 384) and torch.isfinite(z).all()
    assert not z.requires_grad
    encoder.train()
    assert not encoder.backbone.training
    torch.testing.assert_close(z, encoder(images), rtol=0, atol=0)
    result = {"checkpoint_sha256": SHA256, "shape": list(z.shape), "device": str(device), "forward_seconds": elapsed, "frozen": True}
    Path(cfg.output).mkdir(parents=True, exist_ok=True)
    Path(cfg.output, "encoder_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
