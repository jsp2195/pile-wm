"""Seeded stage training. Perception is frozen; decoder never trains dynamics."""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from pile_wm.data import Trajectories
from pile_wm.sim import State, render, state_maps
from pile_wm.models.encoder import DinoEncoder, Standardizer, SHA256
from pile_wm.models.decoder import DensityDecoder, decoder_loss


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def sample_frames(dataset, count, generator):
    xs, ps = [], []
    for _ in range(count):
        trajectory = dataset[int(torch.randint(len(dataset), (), generator=generator))]
        frame = int(torch.randint(len(trajectory["particles"]), (), generator=generator))
        xs.append(trajectory["particles"][frame])
        ps.append(trajectory["pusher"][frame])
    return State(torch.stack(xs), torch.stack(ps))


def encode_states(encoder, state, config, normalizer=None):
    chunks = []
    for start in range(0, len(state.pusher), config.train.encoder_batch_size):
        s = State(state.particles[start:start+config.train.encoder_batch_size], state.pusher[start:start+config.train.encoder_batch_size])
        z = encoder(render(s, config.sim))
        chunks.append(normalizer(z) if normalizer is not None else z)
    return torch.cat(chunks)


def load_perception(config, device):
    encoder = DinoEncoder().to(device)
    saved = torch.load(Path(config.output)/"perception.pt", map_location=device, weights_only=True)
    if saved["encoder_sha256"] != SHA256:
        raise RuntimeError("Checkpoint encoder identity mismatch")
    norm = Standardizer().to(device)
    norm.load_state_dict(saved["normalizer"])
    decoder = DensityDecoder().to(device)
    decoder.load_state_dict(saved["decoder"])
    return encoder, norm, decoder.eval()


def train_decoder(config, device):
    root = Path(config.output)
    root.mkdir(parents=True, exist_ok=True)
    dataset = Trajectories(root/"data", "train")
    validation = Trajectories(root/"data", "val")
    g = torch.Generator().manual_seed(config.seed+200)
    encoder = DinoEncoder().to(device)
    normalizer = Standardizer().to(device)
    started = time.perf_counter()
    statistics = sample_frames(dataset, config.train.statistics_frames, g).to(device)
    # Streaming statistics, sampled exclusively from training trajectory IDs.
    def batches():
        for start in range(0, len(statistics.pusher), config.train.encoder_batch_size):
            s = State(statistics.particles[start:start+config.train.encoder_batch_size], statistics.pusher[start:start+config.train.encoder_batch_size])
            yield encoder(render(s, config.sim))
    count = normalizer.fit(batches())
    decoder = DensityDecoder().to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config.train.decoder_learning_rate)
    log = []
    with SummaryWriter(str(root/"tensorboard")) as writer:
        for step in range(config.train.decoder_steps):
            state = sample_frames(dataset, config.train.batch_size, g).to(device)
            z = encode_states(encoder, state, config, normalizer)
            prediction = decoder(z)
            loss = decoder_loss(prediction, state_maps(state))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            log.append({"step": step, "loss": float(loss.detach())})
            writer.add_scalar("decoder/train_loss", float(loss.detach()), step)
    with torch.no_grad():
        val = sample_frames(validation, max(2, config.train.batch_size), g).to(device)
        val_loss = float(decoder_loss(decoder(encode_states(encoder, val, config, normalizer)), state_maps(val)))
    torch.save({"normalizer": normalizer.state_dict(), "decoder": decoder.state_dict(),
                "encoder_sha256": SHA256, "config": asdict(config), "statistics_split": "train",
                "statistics_frame_count": config.train.statistics_frames}, root/"perception.pt")
    with (root/"decoder_training.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerows(log)
    result = {"config": config.name, "steps": len(log), "last_train_loss": log[-1]["loss"],
              "validation_loss": val_loss, "statistics_tokens": count, "seconds": time.perf_counter()-started}
    save_json(root/"decoder_metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result
