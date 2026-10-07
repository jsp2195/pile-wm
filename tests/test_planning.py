import torch
from torch import nn
from pile_wm.config import PlanConfig, SimConfig
from pile_wm.planning import cem, ModelCost, packed_goal, reachable_goal, gather_fraction
from pile_wm.sim import PileSim


def test_cem_optimizes_seeded_box_problem():
    config = PlanConfig(horizon=3, population=64, elites=8, iterations=5, chunk_size=7)
    target = torch.tensor([.6, -.4])
    cost = lambda a: (a-target).square().mean((1, 2))
    first, value = cem(cost, config, torch.device('cpu'), 2)
    second, _ = cem(cost, config, torch.device('cpu'), 2)
    assert torch.equal(first, second)
    assert value < .015
    assert first.abs().max() <= 1


def test_model_cost_queries_only_autoregressive_model():
    class Toy(nn.Module):
        def forward(self, history, actions):
            return history[:, -1]+actions[:, -1, :1, None]
    cost = ModelCost(Toy(), torch.zeros(1, 3, 4, 8), torch.zeros(1, 2, 2), torch.ones(1, 4, 8))
    sequences = torch.tensor([[[.5, 0.], [.5, 0.]], [[0., 0.], [0., 0.]]])
    torch.testing.assert_close(cost(sequences), torch.tensor([0., 1.]))


def test_task_goals_are_geometric_and_reachable():
    config = SimConfig(particles=20)
    sim = PileSim(config)
    packed, radius = packed_goal(config, 'cpu')
    assert sim.penetration(packed) == 0
    assert gather_fraction(packed, radius) == 1
    initial = sim.reset(1, 8)
    goal, hidden = reachable_goal(sim, initial, 4, 12)
    replay = initial
    for action in hidden:
        replay = sim.step(replay, action)
    torch.testing.assert_close(goal.particles, replay.particles, rtol=0, atol=0)
    torch.testing.assert_close(goal.pusher, replay.pusher, rtol=0, atol=0)
