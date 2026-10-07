"""Frame-causal patch transformer with action conditioning at every token."""
import torch
from torch import nn
import torch.nn.functional as F
from pile_wm.config import ModelConfig


class LatentDynamics(nn.Module):
    def __init__(self, config=None, latent_dim=384, tokens=256):
        super().__init__()
        c = config or ModelConfig()
        self.config, self.tokens = c, tokens
        self.input = nn.Linear(latent_dim, c.dim)
        self.action = nn.Sequential(nn.Linear(2, c.dim), nn.GELU(), nn.Linear(c.dim, c.dim))
        self.spatial = nn.Parameter(torch.randn(1, 1, tokens, c.dim)*.02)
        self.temporal = nn.Parameter(torch.randn(1, c.context, 1, c.dim)*.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(c.dim, c.heads, c.dim*4, dropout=0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(c.depth)])
        self.norm = nn.LayerNorm(c.dim)
        self.output = nn.Linear(c.dim, latent_dim)
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)
        frames = torch.arange(c.context).repeat_interleave(tokens)
        self.register_buffer("mask", frames[None, :] > frames[:, None], persistent=False)

    def predict_all(self, history, actions):
        b, t, p, _ = history.shape
        if (t, p) != (self.config.context, self.tokens) or actions.shape != (b, t, 2):
            raise ValueError("expected three aligned state/action frames")
        x = self.input(history)+self.action(actions)[:, :, None]+self.spatial+self.temporal
        x = x.reshape(b, t*p, -1)
        for layer in self.layers:
            x = layer(x, src_mask=self.mask, is_causal=False)
        delta = self.output(self.norm(x)).reshape(b, t, p, -1)
        return history+delta

    def forward(self, history, actions):
        return self.predict_all(history, actions)[:, -1]


def rollout(model, history, past_actions, future_actions):
    """Only the initial history is observed; each later frame is a prediction.

    history: [B,3,P,C]; past_actions: [B,2,2] connecting those frames;
    future_actions: [B,H,2], starting with the outgoing action of history[-1].
    No detach: multistep training differentiates through the whole rollout.
    """
    outputs = []
    for action in future_actions.unbind(1):
        aligned = torch.cat((past_actions, action[:, None]), 1)
        next_frame = model(history, aligned)
        outputs.append(next_frame)
        history = torch.cat((history[:, 1:], next_frame[:, None]), 1)
        past_actions = aligned[:, 1:]
    if not outputs:
        return history[:, :0]
    return torch.stack(outputs, 1)


def dynamics_loss(model, states, actions, multistep_weight=1.):
    context = model.config.context
    steps = 3
    teacher = torch.stack([model(states[:, i:i+context], actions[:, i:i+context]) for i in range(steps)], 1)
    target = states[:, context:context+steps]
    autonomous = rollout(model, states[:, :context], actions[:, :context-1], actions[:, context-1:context-1+steps])
    one = F.mse_loss(teacher, target)
    multi = F.mse_loss(autonomous, target)
    return one+multistep_weight*multi, one, multi
