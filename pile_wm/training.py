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


def sample_windows(dataset, batch, generator, context=3, future=3):
    xs, ps, actions = [], [], []
    for _ in range(batch):
        trajectory = dataset[int(torch.randint(len(dataset), (), generator=generator))]
        length = context+future
        start = int(torch.randint(len(trajectory["particles"])-length+1, (), generator=generator))
        xs.append(trajectory["particles"][start:start+length])
        ps.append(trajectory["pusher"][start:start+length])
        actions.append(trajectory["actions"][start:start+length-1])
    return State(torch.stack(xs), torch.stack(ps)), torch.stack(actions)


def train_model(config, device):
    from pile_wm.models.dynamics import LatentDynamics, dynamics_loss
    root = Path(config.output)
    encoder, normalizer, _ = load_perception(config, device)
    train = Trajectories(root/"data", "train")
    val = Trajectories(root/"data", "val")
    model = LatentDynamics(config.model).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if not 10_000_000 <= parameters <= 25_000_000:
        raise ValueError(f"experiment model has {parameters} parameters, expected 10–25M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    g = torch.Generator().manual_seed(config.seed+300)
    started = time.perf_counter()
    rows, encoding_seconds, dynamics_seconds = [], 0., 0.
    with SummaryWriter(str(root/"tensorboard")) as writer:
        for step in range(config.train.model_steps):
            state, actions = sample_windows(train, config.train.batch_size, g)
            b, t, n, _ = state.particles.shape
            sync(device)
            before = time.perf_counter()
            z = encode_states(encoder, State(state.particles.reshape(b*t, n, 2), state.pusher.reshape(b*t, 2)).to(device), config, normalizer).reshape(b, t, 256, 384)
            sync(device)
            encoding_seconds += time.perf_counter()-before
            before = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, teacher, multi = dynamics_loss(model, z, actions.to(device), config.train.multistep_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            sync(device)
            dynamics_seconds += time.perf_counter()-before
            row = {"step": step, "loss": float(loss.detach()), "teacher_mse": float(teacher.detach()), "multistep_mse": float(multi.detach())}
            rows.append(row)
            for name, value in row.items():
                if name != "step":
                    writer.add_scalar("dynamics/"+name, value, step)
            print(f"Predictor step {step+1}/{config.train.model_steps}: {row['loss']:.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        state, actions = sample_windows(val, config.train.batch_size, g)
        b, t, n, _ = state.particles.shape
        z = encode_states(encoder, State(state.particles.reshape(b*t, n, 2), state.pusher.reshape(b*t, 2)).to(device), config, normalizer).reshape(b, t, 256, 384)
        val_loss, _, _ = dynamics_loss(model, z, actions.to(device), config.train.multistep_weight)
    torch.save({"model": model.state_dict(), "config": asdict(config), "parameters": parameters,
                "encoder_sha256": SHA256}, root/"dynamics.pt")
    with (root/"model_training.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"config": config.name, "parameters": parameters, "steps": len(rows),
              "last_train_loss": rows[-1]["loss"], "validation_loss": float(val_loss),
              "encoding_seconds": encoding_seconds, "dynamics_seconds": dynamics_seconds,
              "latent_cache_used": False, "seconds": time.perf_counter()-started}
    save_json(root/"training_metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def load_model(config, device):
    from pile_wm.models.dynamics import LatentDynamics
    saved = torch.load(Path(config.output)/"dynamics.pt", map_location=device, weights_only=True)
    if saved["encoder_sha256"] != SHA256 or saved["config"]["model"] != asdict(config.model):
        raise RuntimeError("Dynamics checkpoint/config mismatch")
    model = LatentDynamics(config.model).to(device)
    model.load_state_dict(saved["model"])
    return model.eval()
