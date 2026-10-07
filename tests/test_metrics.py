import pytest
import torch
from pile_wm.metrics import chamfer, correlation, density_iou, mean_se


def test_physical_metrics_have_known_answers():
    x = torch.tensor([[[0., 0.], [1., 0.]]])
    assert float(chamfer(x, x.flip(1))) == 0
    torch.testing.assert_close(chamfer(x, x+torch.tensor([0., 1.])), torch.ones(1))
    a = torch.tensor([[[1., 0.], [0., 1.]]])
    b = torch.tensor([[[1., 0.], [1., 0.]]])
    assert float(density_iou(a, b)) == pytest.approx(1/3)
    assert correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1)
    assert correlation([1, 1, 1], [2, 4, 6]) is None
    summary = mean_se([1, 3])
    assert summary == {"mean": 2., "se": 1., "n": 2}
