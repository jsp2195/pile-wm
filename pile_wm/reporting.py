"""Measured artifacts and README generation; unrun presets remain pending."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/workspace/.cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/workspace/.cache")
Path(os.environ["XDG_CACHE_HOME"], "fontconfig").mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pile_wm.config import load_config
from pile_wm.training import save_json


def create_figures(config, metrics, destination):
    ev, planning = metrics["evaluation"], metrics["planning"]
    h = ev["horizons"]
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120})
    def save(fig, filename):
        fig.suptitle(f"pile-wm · {config.name} · seed {config.seed}")
        fig.tight_layout()
        fig.savefig(destination/filename)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(h, ev["latent_mse"], label="Learned rollout")
    axes[0].plot(h, ev["copy_latent_mse"], label="Copy last frame")
    axes[0].set(xlabel="Horizon", ylabel="Standardized latent MSE")
    axes[1].plot(h, ev["density_iou"], label="Learned rollout")
    axes[1].plot(h, ev["copy_density_iou"], label="Copy last frame")
    axes[1].plot(h, ev["decoder_reconstruction_iou"][1:], "--", label="Decode true latent")
    axes[1].set(xlabel="Horizon", ylabel="Particle density occupancy IoU", ylim=(0, 1))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    save(fig, "rollout_error.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h, ev["latent_mse"], label="True actions")
    ax.plot(h, ev["shuffled_latent_mse"], label="Shuffled actions")
    ax.set(xlabel="Horizon", ylabel="Latent MSE", title=f"Relative shuffle gap: {100*ev['action_shuffle']['relative_gap']:.3f}%")
    ax.legend()
    save(fig, "action_shuffle.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cf = ev["counterfactual"]
    for ax, kind in zip(axes, ("latent", "density")):
        ax.scatter(cf[f"true_{kind}_divergence"], cf[f"predicted_{kind}_divergence"])
        value = cf[f"{kind}_correlation"]
        label = "undefined (zero variance)" if value is None else f"{value:.3f}"
        ax.set(xlabel=f"True pairwise {kind} MSE", ylabel=f"Predicted pairwise {kind} MSE", title=f"{kind}: r = {label}; K={cf['k']}")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))
    save(fig, "counterfactual.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    drift = np.array(ev["mass_drift_by_trajectory"])*100
    for row in drift:
        ax.plot(range(len(row)), row, color="C0", alpha=.25)
    ax.plot(range(drift.shape[1]), drift.mean(0), label="Mean decoded rollout")
    ax.axhline(0, color="black", linestyle="--", label="True mass / copy baseline")
    ax.set(xlabel="Horizon", ylabel="Density mass change from decoded step 0 (%)")
    ax.legend()
    save(fig, "mass_drift.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, task in zip(axes, ("reach", "gather")):
        summary = planning["summary"][task]
        names = list(summary)
        ax.bar(names, [summary[n]["metric"]["mean"] for n in names], yerr=[summary[n]["metric"]["se"] or 0 for n in names], capsize=4)
        ax.set(ylabel="Chamfer (lower better)" if task == "reach" else "Fraction in disc (higher better)", title=f"{task}; n={planning['episodes_per_task_policy']}; bars ± SE")
        ax.tick_params(axis="x", rotation=25)
    save(fig, "planning_comparison.png")
    maps = np.load(Path(config.output)/"rollout_maps.npz")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, key, title in zip(axes, ("truth", "reconstructed", "predicted"), ("Ground truth density", "Decode true DINOv2", "Autoregressive prediction")):
        ax.imshow(maps[key][-1, 0], origin="lower", vmin=0, vmax=1, cmap="magma")
        ax.set_title(title)
        ax.axis("off")
    save(fig, "decoder_reconstruction.png")
    frames = []
    for t in range(len(maps["truth"])):
        canvas = Image.new("RGB", (448, 252), "white")
        for column, key in enumerate(("truth", "predicted")):
            rgb = (plt.get_cmap("magma")(np.clip(maps[key][t, 0], 0, 1))[..., :3]*255).astype(np.uint8)
            pane = Image.fromarray(rgb[::-1]).resize((224, 224), Image.Resampling.NEAREST)
            canvas.paste(pane, (column*224, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 5), f"{config.name}: true density, h={t}", fill="black")
        draw.text((230, 5), "Predicted density", fill="black")
        frames.append(canvas)
    frames[0].save(destination/"rollout_true_vs_predicted.gif", save_all=True, append_images=frames[1:], duration=140, loop=0)
    shutil.copyfile(Path(config.output)/"random_push.gif", destination/"random_push.gif")


def value(x, digits=6):
    return "undefined" if x is None else f"{x:.{digits}g}"


def write_readme():
    reports = {}
    for name in ("smoke", "small", "full"):
        path = Path("reports")/name/"metrics.json"
        if path.exists():
            reports[name] = json.loads(path.read_text())
    lines = ["# pile-wm", "", "An action-conditioned latent dynamics research prototype for granular pile pushing, evaluated by rolling out its own predictions and planning in the real simulator.", "",
             "**Pipeline completion is not evidence of a successful world model.** The measured results below retain failed controls and task outcomes. Unrun configurations are pending.", "", "## Setup", "",
             "Python 3.11, uv, PyTorch, plain dataclass/YAML configuration, pytest, matplotlib, CSV and local TensorBoard. No accounts or external logging services. Run from the checkout root:", "", "```bash",
             "# Cloud machine has a read-only home; these cache directories are writable.", "export UV_CACHE_DIR=/workspace/.cache/uv", "export UV_PYTHON_INSTALL_DIR=/workspace/.local/python",
             "export TORCH_HOME=/workspace/.cache/torch", "export MPLCONFIGDIR=/workspace/.cache/matplotlib", "uv sync --frozen", "uv run --frozen pytest -q", "```", "",
             "The official frozen DINOv2 ViT-S/14 checkpoint is downloaded from `dl.fbaipublicfiles.com` over verified HTTPS and checked against its recorded SHA-256. Download failure stops execution; there is no substitute encoder. Unit tests use tiny fixtures and need no network. Auto device selection is CUDA → MPS → CPU. All experiments are seeded from YAML; deterministic PyTorch algorithms are requested. GPU/MPS behavior remains unverified on this CPU host.", "",
             "## Run the complete pipeline", "", "```bash", "uv run --frozen python -m scripts.pipeline --config configs/smoke.yaml", "uv run --frozen python -m scripts.pipeline --config configs/small.yaml", "# Explicit opt-in only; not run during development:", "uv run --frozen python -m scripts.pipeline --config configs/full.yaml", "```", "",
             "The pipeline runs tests, generates or reuses version/config-matched state-only data, audits every saved state, makes the pushing GIF, trains perception and dynamics, evaluates horizons 1–30, compares planners, and writes every report figure plus `artifacts/<config>/metrics.json`. Add `--regenerate` to force data regeneration. Reuse applies only to simulator states, never cached latents. All training/evaluation stages rerun. The checked-in `reports/<config>/` contains measured metrics and figures; large datasets/checkpoints stay in ignored `artifacts/`. CSV and TensorBoard logs are local.", "",
             "### Budget settings (configuration, not measured performance)", "", "| Config | Trajectories × actions | Predictor updates | Decoder updates | Planning episodes per task/policy | MPC steps |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("smoke", "small", "full"):
        cfg = load_config(f"configs/{name}.yaml")
        lines.append(f"| {name} | {cfg.trajectories} × {cfg.steps} | {cfg.train.model_steps} | {cfg.train.decoder_steps} | {cfg.planning.episodes} | {cfg.planning.steps} |")
    lines += ["", "Smoke is a short CPU integration run, not a trained policy benchmark. Small/full use at least 50 episodes with paired starts and report sample standard errors. No full run is launched implicitly.", "", "## Stage entry points", "", "Use the same `--config` argument for each stage:", "", "```bash"]
    for name in ("random_push", "generate_data", "validate_data", "train_decoder", "train_model", "evaluate", "plan", "report"):
        lines.append(f"uv run --frozen python -m scripts.{name} --config configs/smoke.yaml")
    lines += ["```", "", "`report` redraws the latest complete pipeline metrics. `scripts.check_encoder` is a separate real-checkpoint integration check. `uv run tensorboard --logdir artifacts/smoke/tensorboard` opens local training logs if a desktop is available.", "",
              "## Method and metric definitions", "",
              "The batched PyTorch simulator uses overdamped position-based disc contacts, wall clamping, small pusher substeps, and independently converged environments. Each trajectory stores 200 particle positions, pusher positions, and 40 actions; RGB is rendered on demand. Splits are disjoint trajectory IDs. Correlated random actions reflect momentum at walls.", "",
              "Frozen DINOv2 supplies 16×16×384 patch tokens. Channel statistics use only seeded training-frame samples. A six-layer, 384-wide frame-causal transformer embeds each outgoing action onto every token of its frame, attends freely within that frame, and uses the last three frames. Loss combines teacher-forced latent MSE and a differentiable three-step self-rollout. The separate convolutional decoder detaches its input and predicts 64×64 count density and a pusher heatmap; its gradients cannot train dynamics.", "",
              "Evaluation compares autonomous latent MSE and particle-density IoU against copy-last-frame for every horizon 1–30. Occupancy threshold is fixed at 0.1 count units. Decode-true-latent IoU exposes decoder limitations. Action shuffle uses a cyclic batch derangement and a declared 5% relative error gap. Counterfactuals share one held-out start and branch into K action sequences; latent and particle-density pairwise divergence correlations are separate. Undefined zero-variance correlations are null, never passing. Mass drift is decoded particle mass relative to decoded step 0; true mass is constant.", "",
              "CEM/MPC minimizes final standardized latent MSE to the encoded goal. Learned costs access only predicted latents; oracle costs simulate candidates and encode their final images using the identical cost. Physical scores are computed solely from actual simulator states. Reach uses hidden random sequences and symmetric mean Euclidean Chamfer. Gather uses a nonoverlapping central hexagonal disc packing and the fraction of particle centers in that disc. The scripted gather policy approaches an outer particle from behind and pushes inward. The goal pusher is parked at (0.5,0.95), so latent cost also reflects pusher pose. Reach success requires Chamfer ≤ particle radius and at least 50% improvement from the start; gather success requires fraction ≥ 0.8. Success thresholds do not alter the reported raw metrics.", "",
              "## Measured results", "", "| Config | Status | Tests | Total seconds | Data reused | World-model competence demonstrated |", "| --- | --- | ---: | ---: | --- | --- |"]
    for name in ("smoke", "small", "full"):
        if name not in reports:
            lines.append(f"| {name} | Pending; unrun | — | — | — | Unverified |")
        else:
            m = reports[name]
            lines.append(f"| {name} | Complete pipeline | {m['tests_passed']} | {value(m.get('total_seconds', m['seconds_before_reporting']))} | {m['dataset_reused']} | {m['world_model_demonstrated']} |")
    for name, m in reports.items():
        ev, planning = m["evaluation"], m["planning"]
        if m.get("previous_runs"):
            lines += ["", f"Previous completed **{name}** runs (same resolved configuration):", "",
                      "| Total seconds | Data reused | Tests passed |", "| ---: | --- | ---: |"]
            for run in m["previous_runs"]:
                lines.append(f"| {value(run['total_seconds'])} | {run['dataset_reused']} | {run['tests_passed']} |")
        lines += ["", f"### {name}", "", f"[Raw metrics](reports/{name}/metrics.json), including resolved configuration, source hashes, stage timing, per-episode outcomes and standard errors.", "",
                  f"Mean horizon-1–30 latent MSE: learned **{value(np.mean(ev['latent_mse']))}**, copy **{value(np.mean(ev['copy_latent_mse']))}**. Beats copy: **{ev['beats_copy_mean_mse']}**. Action-shuffle relative gap: **{value(100*ev['action_shuffle']['relative_gap'])}%**; passes declared threshold: **{ev['action_shuffle']['passes']}**.", "",
                  f"Counterfactual correlation: latent **{value(ev['counterfactual']['latent_correlation'])}**, particle density **{value(ev['counterfactual']['density_correlation'])}**. Final-horizon density IoU: **{value(ev['density_iou'][-1])}**, decode-true-latent IoU: **{value(ev['decoder_reconstruction_iou'][-1])}**. Final decoded mass drift: **{value(100*ev['mass_drift_relative'][-1])}%**.", "",
                  f"Simulator audit: {m['data']['transitions']} transitions, maximum penetration {value(m['data']['max_penetration'])} against tolerance {value(m['data']['penetration_tolerance'])}. Predictor parameters: {m['training']['parameters']:,}. Online encoding took {value(m['training']['encoding_seconds'])} seconds versus {value(m['training']['dynamics_seconds'])} seconds for predictor optimization; latent cache used: {m['training']['latent_cache_used']}.", "",
                  "| Task | Policy | True-state metric ± SE | Seconds/episode ± SE | Success rate |", "| --- | --- | ---: | ---: | ---: |"]
        for task, summaries in planning["summary"].items():
            for policy, row in summaries.items():
                metric, seconds = row["metric"], row["seconds"]
                lines.append(f"| {task} | {policy} | {value(metric['mean'])} ± {value(metric['se'])} | {value(seconds['mean'])} ± {value(seconds['se'])} | {value(row['success_rate']['mean'])} |")
        lines += ["", f"All rows above use **{name}**, {planning['episodes_per_task_policy']} episodes per task/policy. Reach is lower-better; gather is higher-better. Episode times include controller queries, observations and real steps, but exclude shared goal construction and model loading. Planning advantage demonstrated: **{planning['planning_advantage_demonstrated']}**.", ""]
        for image, label in (("rollout_error.png", "Rollout error and density IoU"), ("action_shuffle.png", "Action shuffle"), ("counterfactual.png", "Counterfactual divergence"), ("mass_drift.png", "Mass drift"), ("planning_comparison.png", "Planning comparison"), ("decoder_reconstruction.png", "Decoder reconstruction control"), ("rollout_true_vs_predicted.gif", "True versus predicted density rollout"), ("random_push.gif", "Random pushing")):
            lines += [f"![{name}: {label}](reports/{name}/{image})", ""]
    lines += ["## Interpretation and next work", "", "The short smoke run is not evidence that the model qualifies as a successful world model. Check the copy baseline, action shuffle, counterfactual particle-density correlation, decoder reconstruction, and physical planning outcomes together. Tiny reach motions make no-op competitive; two smoke episodes do not establish a planning advantage. Finite CEM budgets and latent/physical cost mismatch can also limit the oracle. No metric or threshold was tuned to conceal a failure.", "",
              "Next: run the documented small training budget on a GPU, improve decoder reconstruction and action-sensitive dynamics using training/validation data, then rerun unchanged held-out metrics and all paired planner baselines. GPU/MPS execution, small/full efficacy, and 50-episode uncertainty estimates remain unverified until those runs exist. Conditional flow matching is intentionally deferred until milestones 1–6 pass the empirical checks.", "",
              "See [DECISIONS.md](DECISIONS.md) for choices and failed experiments, and [AGENTS.md](AGENTS.md) for repository conventions."]
    Path("README.md").write_text("\n".join(lines)+"\n")


def create_report(config, metrics=None, figures=True):
    if metrics is None:
        metrics = json.loads((Path(config.output)/"metrics.json").read_text())
    destination = Path("reports")/config.name
    destination.mkdir(parents=True, exist_ok=True)
    if figures:
        create_figures(config, metrics, destination)
    save_json(destination/"metrics.json", metrics)
    write_readme()
