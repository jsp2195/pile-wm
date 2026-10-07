import torch
from pile_wm.config import ModelConfig
from pile_wm.models.dynamics import LatentDynamics, rollout, dynamics_loss


def tiny():
    torch.manual_seed(1)
    return LatentDynamics(ModelConfig(dim=16, depth=1, heads=2), latent_dim=8, tokens=4).eval()


def test_frame_causality_and_action_conditioning():
    model = tiny()
    history, actions = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 2)
    reference = model.predict_all(history, actions)
    changed, other = history.clone(), actions.clone()
    changed[:, 2] += 100
    other[:, 2] -= 20
    torch.testing.assert_close(model.predict_all(changed, other)[:, :2], reference[:, :2], atol=0, rtol=0)
    assert not torch.allclose(model(history, other), model(history, actions))
    # Within a frame, changing one token can affect another token's prediction.
    changed = history.clone()
    changed[:, 0, 0] += torch.arange(8)
    assert not torch.allclose(model.predict_all(changed, actions)[:, 0, 1], reference[:, 0, 1])


def test_autoregression_consumes_predictions_and_retains_gradients():
    from torch import nn
    class AddAction(nn.Module):
        def forward(self, history, actions):
            return history[:, -1]+actions[:, -1, :1, None]
    start = torch.zeros(1, 3, 4, 8, requires_grad=True)
    future = torch.tensor([[[1., 0.], [2., 0.], [-1., 0.]]])
    predicted = rollout(AddAction(), start, torch.zeros(1, 2, 2), future)
    torch.testing.assert_close(predicted[0, :, 0, 0], torch.tensor([1., 3., 2.]))
    predicted[:, -1].sum().backward()
    assert start.grad[:, -1].abs().sum() > 0


def test_multistep_loss_backpropagates_and_parameter_budget():
    model = tiny()
    states = torch.randn(2, 6, 4, 8)
    actions = torch.randn(2, 5, 2)
    loss, teacher, multi = dynamics_loss(model, states, actions)
    torch.testing.assert_close(loss, teacher+multi)
    loss.backward()
    assert model.action[0].weight.grad.abs().sum() > 0
    real = LatentDynamics()
    assert 10_000_000 <= sum(p.numel() for p in real.parameters()) <= 25_000_000


def test_evaluation_fastpath_preserves_frame_causality():
    model = tiny()
    history, actions = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 2)
    with torch.inference_mode():
        original = model.predict_all(history, actions)
        history[:, -1] *= 100
        actions[:, -1] *= -100
        changed = model.predict_all(history, actions)
    torch.testing.assert_close(original[:, :2], changed[:, :2], rtol=0, atol=0)
