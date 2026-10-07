import torch
from pile_wm.config import Config, SimConfig
from pile_wm.data import generate, split_ids, Trajectories


def test_split_is_disjoint_complete_seeded():
    a = split_ids(32, 42)
    assert a == split_ids(32, 42)
    assert {k: len(v) for k, v in a.items()} == {"train": 26, "val": 3, "test": 3}
    flat = sum(a.values(), [])
    assert sorted(flat) == list(range(32))
    assert a != split_ids(32, 43)


def test_state_only_generation_roundtrip_and_reproducibility(tmp_path):
    cfg = Config(trajectories=5, steps=4, data_batch_size=3,
                 sim=SimConfig(particles=8, contact_iterations=4, init_iterations=10))
    generate(cfg, "cpu", tmp_path/"a")
    generate(cfg, "cpu", tmp_path/"b")
    ds = Trajectories(tmp_path/"a", "train")
    other = Trajectories(tmp_path/"b", "train")
    for i in range(len(ds)):
        item = ds[i]
        assert set(item) == {"particles", "pusher", "actions"}
        for key in item:
            assert torch.equal(item[key], other[i][key])
        assert item["particles"].shape == (5, 8, 2)
        assert item["actions"].shape == (4, 2)
        assert item["actions"].abs().max() <= 1
    assert Trajectories.images(ds[0], cfg.sim).shape == (5, 3, 224, 224)


def test_validate_dataset_detects_corruption(tmp_path):
    from scripts.validate_data import validate
    import pytest
    cfg = Config(output=str(tmp_path), trajectories=5, steps=4, data_batch_size=5,
                 sim=SimConfig(particles=8, contact_iterations=10, init_iterations=100))
    generate(cfg, "cpu")
    assert validate(cfg)["geometry_passed"]
    path = tmp_path/"data"/"trajectory_00000.pt"
    item = torch.load(path, weights_only=True)
    item["particles"][0, 0, 0] = -1
    torch.save(item, path)
    with pytest.raises(AssertionError):
        validate(cfg)
