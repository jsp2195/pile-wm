import torch
from torch import nn
from pile_wm.models.encoder import Standardizer
from pile_wm.models.decoder import DensityDecoder, decoder_loss


class TinyEncoder(nn.Module):
    """Network-free unit fixture; never exposed by experiment configuration."""
    dim = 8
    tokens = 256

    def forward(self, images):
        small = torch.nn.functional.adaptive_avg_pool2d(images, (16, 16))
        return small.flatten(2).transpose(1, 2).mean(-1, keepdim=True).expand(-1, -1, self.dim)


def test_standardization_training_statistics():
    g = torch.Generator().manual_seed(7)
    train = torch.randn(5, 256, 8, generator=g)*2+3
    standardizer = Standardizer(8)
    assert standardizer.fit([train[:2], train[2:]]) == 5*256
    normalized = standardizer(train)
    torch.testing.assert_close(normalized.mean((0, 1)), torch.zeros(8), atol=2e-6, rtol=0)
    torch.testing.assert_close(normalized.std((0, 1), unbiased=False), torch.ones(8))
    mean = standardizer.mean.clone()
    standardizer(torch.ones_like(train)*100)
    assert torch.equal(mean, standardizer.mean)


def test_decoder_has_no_predictor_gradient():
    predictor = nn.Linear(8, 8)
    z = predictor(TinyEncoder()(torch.rand(2, 3, 224, 224)))
    decoder = DensityDecoder(8, width=8)
    result = decoder(z)
    assert result.shape == (2, 2, 64, 64)
    target = torch.zeros_like(result)
    target[:, 0, 20:30, 20:30] = 2
    target[:, 1, 30, 30] = 1
    decoder_loss(result, target).backward()
    assert all(p.grad is None for p in predictor.parameters())
    assert any(p.grad is not None for p in decoder.parameters())
    assert (result > 0).all()
