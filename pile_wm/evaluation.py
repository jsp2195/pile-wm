"""Held-out rollouts and interventions, including copy and shuffled-action controls."""
from pathlib import Path
import time
import numpy as np
import torch
from pile_wm.data import Trajectories
from pile_wm.models.dynamics import rollout
from pile_wm.sim import PileSim, State, state_maps
from pile_wm.training import load_perception, load_model, encode_states, save_json, sync
from pile_wm.metrics import density_iou, correlation, mean_se


@torch.no_grad()
def evaluate(config, device):
    root = Path(config.output)
    encoder, norm, decoder = load_perception(config, device)
    model = load_model(config, device)
    data = Trajectories(root/"data", "test")
    b = min(config.evaluation.trajectories, len(data))
    if b < 2:
        raise ValueError("action-shuffle evaluation requires at least two held-out trajectories")
    horizon = config.evaluation.horizon
    if horizon != 30 or config.steps < horizon+2:
        raise ValueError("evaluation must cover horizons 1 through 30")
    started = time.perf_counter()
    items = [data[i] for i in range(b)]
    x = torch.stack([d["particles"][:horizon+3] for d in items]).to(device)
    p = torch.stack([d["pusher"][:horizon+3] for d in items]).to(device)
    actions = torch.stack([d["actions"][:horizon+2] for d in items]).to(device)
    z = encode_states(encoder, State(x.flatten(0, 1), p.flatten(0, 1)), config, norm).reshape(b, horizon+3, 256, 384)
    predicted = rollout(model, z[:, :3], actions[:, :2], actions[:, 2:])
    # A cyclic derangement has no accidentally unchanged action sequence.
    permutation = torch.arange(b, device=device).roll(1)
    shuffled = rollout(model, z[:, :3], actions[:, :2], actions[permutation, 2:])
    truth = z[:, 3:]
    copy = z[:, 2:3].expand_as(truth)
    mse = (predicted-truth).square().mean((-1, -2))
    baseline = (copy-truth).square().mean((-1, -2))
    shuffle_error = (shuffled-truth).square().mean((-1, -2))
    # Decode in small batches, without retaining predictor computation graphs.
    def decode(sequence):
        flat = sequence.flatten(0, 1)
        return torch.cat([decoder(flat[i:i+config.train.encoder_batch_size]) for i in range(0, len(flat), config.train.encoder_batch_size)]).reshape(b, -1, 2, 64, 64)
    decoded = decode(torch.cat((z[:, 2:3], predicted), 1))
    reconstructed = decode(z[:, 2:])
    physical = state_maps(State(x[:, 2:].flatten(0, 1), p[:, 2:].flatten(0, 1))).reshape(b, horizon+1, 2, 64, 64)
    pred_iou = density_iou(decoded[:, 1:, 0], physical[:, 1:, 0], config.evaluation.occupancy_threshold)
    copy_iou = density_iou(decoded[:, :1, 0].expand(-1, horizon, -1, -1), physical[:, 1:, 0], config.evaluation.occupancy_threshold)
    reconstruction_iou = density_iou(reconstructed[:, :, 0], physical[:, :, 0], config.evaluation.occupancy_threshold)
    mass = decoded[:, :, 0].sum((-1, -2))
    drift = mass/mass[:, :1].clamp_min(1e-8)-1
    gap = float((shuffle_error.mean()-mse.mean())/mse.mean().clamp_min(1e-12))
    paired_gap = (shuffle_error-mse).mean(1)

    k, ch = config.evaluation.counterfactual_k, config.evaluation.counterfactual_horizon
    if k < 3:
        raise ValueError("counterfactual evaluation requires K >= 3")
    g = torch.Generator().manual_seed(config.seed+400)
    cf_actions = torch.randn(k, ch, 2, generator=g).to(device)
    for t in range(1, ch):
        cf_actions[:, t] += .85*cf_actions[:, t-1]
    cf_actions.clamp_(-1, 1)
    sim = PileSim(config.sim, device)
    state = State(x[0, 2:3].expand(k, -1, -1).clone(), p[0, 2:3].expand(k, -1).clone())
    cf_predictions = rollout(model, z[0:1, :3].expand(k, -1, -1, -1), actions[0:1, :2].expand(k, -1, -1), cf_actions)
    for action in cf_actions.unbind(1):
        state = sim.step(state, action)
    cf_truth = encode_states(encoder, state, config, norm)
    cf_density = decoder(cf_predictions[:, -1])[:, 0]
    real_density = state_maps(state)[:, 0]
    pairs = torch.triu_indices(k, k, 1, device=device)
    def divergences(v):
        return (v[pairs[0]]-v[pairs[1]]).square().flatten(1).mean(1)
    pred_div = divergences(cf_predictions[:, -1])
    true_div = divergences(cf_truth)
    pred_physical_div, true_physical_div = divergences(cf_density), divergences(real_density)
    counter = {"k": k, "horizon": ch, "start_test_trajectory_id": data.ids[0], "start_frame": 2,
               "predicted_latent_divergence": pred_div.cpu().tolist(), "true_latent_divergence": true_div.cpu().tolist(),
               "latent_correlation": correlation(pred_div.cpu(), true_div.cpu()),
               "predicted_density_divergence": pred_physical_div.cpu().tolist(),
               "true_density_divergence": true_physical_div.cpu().tolist(),
               "density_correlation": correlation(pred_physical_div.cpu(), true_physical_div.cpu()),
               "undefined_correlation_policy": "null when either divergence has zero variance; never counted as passing"}
    sync(device)
    result = {"config": config.name, "test_trajectory_ids": data.ids[:b], "horizons": list(range(1, horizon+1)),
              "latent_mse": mse.mean(0).cpu().tolist(), "copy_latent_mse": baseline.mean(0).cpu().tolist(),
              "shuffled_latent_mse": shuffle_error.mean(0).cpu().tolist(),
              "density_iou": pred_iou.mean(0).cpu().tolist(), "copy_density_iou": copy_iou.mean(0).cpu().tolist(),
              "decoder_reconstruction_iou": reconstruction_iou.mean(0).cpu().tolist(),
              "mass_drift_relative": drift.mean(0).cpu().tolist(),
              "mass_drift_by_trajectory": drift.cpu().tolist(),
              "decoded_ground_truth_mass": reconstructed[:, :, 0].sum((-1, -2)).mean(0).cpu().tolist(),
              "true_mass": config.sim.particles,
              "action_shuffle": {"relative_gap": gap, "paired_absolute_gap": mean_se(paired_gap.cpu().tolist()),
                                  "required_relative_gap": config.evaluation.shuffle_gap_threshold,
                                  "passes": gap > config.evaluation.shuffle_gap_threshold},
              "counterfactual": counter, "beats_copy_mean_mse": bool(mse.mean() < baseline.mean()),
              "seconds": time.perf_counter()-started}
    save_json(root/"evaluation_metrics.json", result)
    np.savez_compressed(root/"rollout_maps.npz", truth=physical[0].cpu().numpy(), predicted=decoded[0].cpu().numpy(),
                        reconstructed=reconstructed[0].cpu().numpy(), copy=decoded[0, :1].expand(horizon+1, -1, -1, -1).cpu().numpy())
    print({"beats_copy": result["beats_copy_mean_mse"], "shuffle_relative_gap": gap,
           "counterfactual_latent_correlation": counter["latent_correlation"], "seconds": result["seconds"]}, flush=True)
    return result
