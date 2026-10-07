"""CEM/MPC. Learned planning queries latent dynamics only; scores use true state."""
from pathlib import Path
import math
import time
import torch
from pile_wm.sim import PileSim, State, render
from pile_wm.models.dynamics import rollout
from pile_wm.training import load_perception, load_model, encode_states, save_json, sync
from pile_wm.metrics import chamfer, mean_se


@torch.no_grad()
def cem(cost, config, device, seed, mean=None):
    if not 1 <= config.elites <= config.population:
        raise ValueError("CEM elite count must be within population")
    g = torch.Generator().manual_seed(seed)
    mean = torch.zeros(config.horizon, 2, device=device) if mean is None else mean.clone()
    std = torch.ones_like(mean)
    best_cost, best = float("inf"), mean.clone()
    for _ in range(config.iterations):
        noise = torch.randn(config.population, config.horizon, 2, generator=g).to(device)
        candidates = (mean[None]+std[None]*noise).clamp(-1, 1)
        candidates[0] = mean
        if config.population > 1:
            candidates[1] = 0  # an explicit no-op candidate in both CEM models
        values = torch.cat([cost(candidates[i:i+config.chunk_size]) for i in range(0, config.population, config.chunk_size)])
        if values.shape != (config.population,) or not torch.isfinite(values).all():
            raise RuntimeError("CEM received invalid candidate costs")
        indices = values.argsort()[:config.elites]
        elite = candidates[indices]
        mean = elite.mean(0)
        std = elite.std(0, unbiased=False).clamp_min(.1)
        if float(values[indices[0]]) < best_cost:
            best_cost = float(values[indices[0]])
            best = candidates[indices[0]].clone()
    return best, best_cost


class ModelCost:
    """Intentionally has no simulator or physical-state dependency."""
    def __init__(self, model, history, past_actions, goal):
        self.model, self.history, self.past_actions, self.goal = model, history, past_actions, goal

    @torch.no_grad()
    def __call__(self, sequences):
        b = len(sequences)
        predicted = rollout(self.model, self.history.expand(b, -1, -1, -1), self.past_actions.expand(b, -1, -1), sequences)
        return (predicted[:, -1]-self.goal).square().mean((-1, -2))


class OracleCost:
    """Same action proposals and latent terminal cost, exact simulator transitions."""
    def __init__(self, sim, state, encoder, normalizer, goal, config):
        self.sim, self.state, self.encoder = sim, state, encoder
        self.normalizer, self.goal, self.config = normalizer, goal, config

    @torch.no_grad()
    def __call__(self, sequences):
        b = len(sequences)
        state = State(self.state.particles.expand(b, -1, -1).clone(), self.state.pusher.expand(b, -1).clone())
        for action in sequences.unbind(1):
            state = self.sim.step(state, action)
        z = encode_states(self.encoder, state, self.config, self.normalizer)
        return (z-self.goal).square().mean((-1, -2))


def packed_goal(config, device):
    """Centered hexagonal packing with enough clearance for all N discs."""
    r, n = config.particle_radius, config.particles
    spacing = 2*r*1.02
    bound = math.ceil(math.sqrt(n))+2
    points = []
    for row in range(-bound, bound+1):
        for column in range(-bound, bound+1):
            points.append([spacing*(column+.5*(row % 2)), spacing*math.sqrt(3)/2*row])
    points = torch.tensor(points, device=device)
    points = points[points.square().sum(-1).argsort()[:n]]
    radius = float(points.norm(dim=-1).max())+r
    if radius >= .45:
        raise ValueError("requested particle packing does not fit table")
    return State(points[None]+.5, torch.tensor([[.5, .95]], device=device)), radius


def gather_fraction(state, radius):
    return ((state.particles-.5).norm(dim=-1) <= radius).float().mean(-1)


def gather_heuristic(state, config):
    radial = state.particles[0]-.5
    index = radial.norm(dim=-1).argmax()
    point = state.particles[0, index]
    direction = radial[index]/radial[index].norm().clamp_min(1e-8)
    behind = (point+direction*(config.pusher_radius+config.particle_radius+.01)).clamp(config.pusher_radius, 1-config.pusher_radius)
    close = (state.pusher[0]-behind).norm() < .025
    target = point-.08*direction if close else behind
    return ((target-state.pusher)/config.max_displacement).clamp(-1, 1)


@torch.no_grad()
def reachable_goal(sim, state, steps, seed):
    g = torch.Generator().manual_seed(seed)
    actions = torch.randn(steps, 1, 2, generator=g).to(state.particles.device)
    for t in range(1, steps):
        actions[t] += .85*actions[t-1]
    actions.clamp_(-1, 1)
    goal = state.clone()
    for action in actions:
        goal = sim.step(goal, action)
    return goal, actions


@torch.no_grad()
def run_planning(config, device):
    if config.name != "smoke" and config.planning.episodes < 50:
        raise ValueError("non-smoke planning requires at least 50 episodes")
    root = Path(config.output)
    encoder, normalizer, _ = load_perception(config, device)
    model = load_model(config, device)
    sim = PileSim(config.sim, device)
    packed, radius = packed_goal(config.sim, device)
    records = []
    started = time.perf_counter()
    for task in ("reach", "gather"):
        policies = ["random", "noop", "learned", "oracle"]
        if task == "gather":
            policies.insert(2, "heuristic")
        for episode in range(config.planning.episodes):
            episode_seed = config.seed+10_000+episode
            initial = sim.reset(1, episode_seed)
            goal, hidden = reachable_goal(sim, initial, config.planning.steps, episode_seed+1000) if task == "reach" else (packed, None)
            goal_latent = encode_states(encoder, goal, config, normalizer)
            for policy in policies:
                state = initial.clone()
                sync(device)
                policy_started = time.perf_counter()
                previous = torch.zeros(1, 2, device=device)
                history, past = None, torch.zeros(1, 2, 2, device=device)
                if policy == "learned":
                    observed = encode_states(encoder, state, config, normalizer)
                    history = observed[:, None].expand(-1, 3, -1, -1).clone()
                warm = None
                g = torch.Generator().manual_seed(episode_seed+2000)
                path = [state.particles.cpu()]
                pushes = [state.pusher.cpu()]
                chosen = []
                for step in range(config.planning.steps):
                    if policy == "noop":
                        action = torch.zeros(1, 2, device=device)
                    elif policy == "random":
                        previous = (.85*previous+.45*torch.randn(1, 2, generator=g).to(device)).clamp(-1, 1)
                        action = previous
                    elif policy == "heuristic":
                        action = gather_heuristic(state, config.sim)
                    else:
                        evaluator = ModelCost(model, history, past, goal_latent) if policy == "learned" else OracleCost(sim, state, encoder, normalizer, goal_latent, config)
                        sequence, _ = cem(evaluator, config.planning, device, episode_seed+3000+step, warm)
                        action = sequence[:1]
                        warm = torch.cat((sequence[1:], torch.zeros_like(sequence[:1])))
                    # This is the only real transition used by the learned controller.
                    state = sim.step(state, action)
                    if policy == "learned" and step+1 < config.planning.steps:
                        observed = encode_states(encoder, state, config, normalizer)
                        history = torch.cat((history[:, 1:], observed[:, None]), 1)
                        past = torch.cat((past[:, 1:], action[:, None]), 1)
                    path.append(state.particles.cpu())
                    pushes.append(state.pusher.cpu())
                    chosen.append(action.cpu())
                sync(device)
                elapsed = time.perf_counter()-policy_started
                metric = float(chamfer(state.particles, goal.particles)) if task == "reach" else float(gather_fraction(state, radius))
                initial_metric = float(chamfer(initial.particles, goal.particles)) if task == "reach" else float(gather_fraction(initial, radius))
                # Reach success also requires 50% reduction from the initial error,
                # preventing nearly unchanged piles from trivially proving success.
                success = metric <= config.sim.particle_radius and metric < .5*initial_metric if task == "reach" else metric >= .8
                records.append({"task": task, "episode": episode, "seed": episode_seed, "policy": policy,
                                "metric": metric, "initial_metric": initial_metric, "success": bool(success), "seconds": elapsed})
                if episode == 0:
                    torch.save({"particles": torch.cat(path), "pusher": torch.cat(pushes), "actions": torch.cat(chosen),
                                "goal_particles": goal.particles.cpu(), "goal_pusher": goal.pusher.cpu()}, root/f"planning_{task}_{policy}.pt")
                print(f"{task} episode {episode+1}/{config.planning.episodes} {policy}: {metric:.6f} ({elapsed:.2f}s)", flush=True)
    summary = {}
    for task in ("reach", "gather"):
        summary[task] = {}
        for policy in sorted({r["policy"] for r in records if r["task"] == task}):
            rows = [r for r in records if r["task"] == task and r["policy"] == policy]
            summary[task][policy] = {"metric": mean_se([r["metric"] for r in rows]),
                                     "seconds": mean_se([r["seconds"] for r in rows]),
                                     "success_rate": mean_se([float(r["success"]) for r in rows])}
    comparisons = {}
    for task in summary:
        sign = -1 if task == "reach" else 1
        comparisons[task] = {}
        for baseline in ("random", "noop"):
            gains = []
            for ep in range(config.planning.episodes):
                learned = next(r["metric"] for r in records if r["task"] == task and r["episode"] == ep and r["policy"] == "learned")
                control = next(r["metric"] for r in records if r["task"] == task and r["episode"] == ep and r["policy"] == baseline)
                gains.append(sign*(learned-control))
            comparisons[task][baseline] = mean_se(gains)
    demonstrated = config.planning.episodes >= 50 and all(v["mean"] > 2*(v["se"] or 0) for t in comparisons.values() for v in t.values())
    result = {"config": config.name, "episodes_per_task_policy": config.planning.episodes,
              "gather_radius": radius, "reach_metric": "symmetric mean Euclidean Chamfer, lower is better",
              "gather_metric": "fraction of particle centers inside central disc, higher is better",
              "success_definition": {"reach": "Chamfer <= particle radius and < 50% of initial Chamfer", "gather": "fraction >= 0.8"},
              "cem_cost": "mean squared standardized DINOv2 final patch tokens vs encoded goal, identical for learned and oracle",
              "timing": "controller including real steps, online observations, and cost evaluations; excludes shared goal construction/model loading",
              "summary": summary, "episodes": records, "paired_learned_gain": comparisons,
              "planning_advantage_demonstrated": demonstrated, "seconds": time.perf_counter()-started}
    save_json(root/"planning_metrics.json", result)
    return result
