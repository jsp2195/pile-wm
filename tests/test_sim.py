import torch
from pile_wm.config import SimConfig
from pile_wm.sim import PileSim, State, render, state_maps


def test_contacts_bounds_count_and_seed():
    sim = PileSim()
    a = sim.reset(2, seed=19)
    b = sim.reset(2, seed=19)
    assert torch.equal(a.particles, b.particles)
    g = torch.Generator().manual_seed(1)
    for _ in range(8):
        action = torch.rand(2, 2, generator=g)*2-1
        a, b = sim.step(a, action), sim.step(b, action)
        assert torch.equal(a.particles, b.particles)
        assert a.particles.shape == (2, 200, 2)
        assert a.particles.min() >= sim.config.particle_radius
        assert a.particles.max() <= 1-sim.config.particle_radius
        assert sim.penetration(a) < sim.config.tolerance


def test_batched_matches_single():
    sim = PileSim(SimConfig(particles=40))
    state = sim.reset(3, 2)
    actions = torch.tensor([[1., 0.], [0., -1.], [-0.5, 0.5]])
    batch = sim.step(state, actions)
    for i in range(3):
        single = sim.step(State(state.particles[i:i+1], state.pusher[i:i+1]), actions[i:i+1])
        torch.testing.assert_close(batch.particles[i:i+1], single.particles, rtol=0, atol=1e-7)


def test_push_moves_particles_and_noop_settled():
    c = SimConfig(particles=1)
    sim = PileSim(c)
    state = State(torch.tensor([[[0.5, 0.5]]]), torch.tensor([[0.44, 0.5]]))
    moved = sim.step(state, torch.tensor([[1., 0.]]))
    assert moved.particles[0, 0, 0] > 0.525
    assert sim.penetration(moved) < c.tolerance
    still = sim.step(state, torch.zeros(1, 2))
    torch.testing.assert_close(still.particles, state.particles)


def test_render_maps():
    sim = PileSim(SimConfig(particles=20))
    state = sim.reset(2)
    image = render(state, sim.config)
    assert image.shape == (2, 3, 224, 224)
    assert image.min() >= 0 and image.max() <= 1
    maps = state_maps(state)
    torch.testing.assert_close(maps[:, 0].sum((1, 2)), torch.full((2,), 20.))
    torch.testing.assert_close(maps[:, 1].sum((1, 2)), torch.ones(2))
