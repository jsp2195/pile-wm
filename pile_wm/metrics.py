"""Metric definitions shared by reporting and tests; no fitted thresholds."""
import math
import torch


def density_iou(predicted, truth, threshold=.1):
    p, t = predicted > threshold, truth > threshold
    return (p & t).sum((-1, -2))/(p | t).sum((-1, -2)).clamp_min(1)


def chamfer(x, y):
    """Symmetric mean nearest-neighbor Euclidean distance, in table units."""
    distance = torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist")
    return .5*(distance.amin(-1).mean(-1)+distance.amin(-2).mean(-1))


def correlation(x, y):
    x, y = torch.as_tensor(x).double(), torch.as_tensor(y).double()
    x, y = x-x.mean(), y-y.mean()
    if len(x) < 2 or x.norm() < 1e-12 or y.norm() < 1e-12:
        return None
    return float((x*y).sum()/(x.norm()*y.norm()))


def mean_se(values):
    x = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(x.mean()), "se": float(x.std(unbiased=True)/math.sqrt(len(x))) if len(x) > 1 else None, "n": len(x)}
