import json

import pytest
import torch

from scripts.asism_v2.contracts import (fingerprint, validate_cache, validate_measurements,
                                       validate_roles, write_new_json)
from scripts.asism_v2.models import SizeAwareSetUtility
from scripts.asism_v2.preserve import sha256, snapshot


def test_exclusive_output(tmp_path):
    path = tmp_path / "result.json"
    write_new_json(path, {"v": 1})
    with pytest.raises(FileExistsError):
        write_new_json(path, {"v": 2})
    assert json.loads(path.read_text()) == {"v": 1}


def test_snapshot_refuses_existing_destination(tmp_path):
    with pytest.raises(FileExistsError):
        snapshot(tmp_path, tmp_path / "source", tmp_path)


@pytest.mark.parametrize("roles", [{"final_eval_heldout": ["a"]},
    {"utility_train": ["a"], "utility_test": ["a"]}, {"utility_train": ["a", "a"]}])
def test_split_leakage_rejected(roles):
    with pytest.raises(ValueError):
        validate_roles(roles)


def test_cache_binds_content_and_configuration(tmp_path):
    path = tmp_path / "predictions"
    path.write_bytes(b"first")
    expected = {"model": "abc", "split": "def", "recipe": "ghi"}
    metadata = {**expected, "payload_sha256": sha256(path)}
    validate_cache(metadata, expected, path)
    with pytest.raises(ValueError):
        validate_cache(metadata, {**expected, "model": "changed"}, path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        validate_cache(metadata, expected, path)


def test_repetition_grid_and_stale_labels():
    protocol, subsets, seeds = {"recipe": "fixed"}, {"s1": ["i1", "i2"]}, [1, 2]
    rows = [{"subset_id": "s1", "seed": seed, "protocol_sha256": fingerprint(protocol),
             "members_sha256": fingerprint({"ids": ["i1", "i2"]}),
             "augmented_metric": .7, "real_metric": .6} for seed in seeds]
    assert validate_measurements(rows, subsets, seeds, protocol)["s1"] == pytest.approx([.7, .7])
    for bad in (rows[:1], rows + rows[:1]):
        with pytest.raises(ValueError):
            validate_measurements(bad, subsets, seeds, protocol)
    with pytest.raises(ValueError):
        validate_measurements(rows, subsets, seeds, {"recipe": "changed"})


def test_size_composition_permutation_and_padding():
    torch.manual_seed(27)
    model = SizeAwareSetUtility(2)
    x = torch.tensor([[[1., 2.], [3., 4.], [float("nan"), 0.]]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    output = model(x, mask)
    assert torch.allclose(output, model(x[:, [1, 2, 0]], mask[:, [1, 2, 0]]), atol=1e-6)
    output.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.isfinite(x.grad).all()
    one = torch.ones(1, 1, 2)
    two = torch.ones(1, 2, 2)
    a = model.encode_set(one, torch.ones(1, 1, dtype=torch.bool))
    b = model.encode_set(two, torch.ones(1, 2, dtype=torch.bool))
    assert torch.allclose(a[:, :-1], b[:, :-1])
    assert not torch.allclose(a[:, -1], b[:, -1])
    assert not torch.allclose(model.encode_set(two * 2, mask[:, :2]), b)


def test_empty_sets_rejected():
    with pytest.raises(ValueError):
        SizeAwareSetUtility(2)(torch.zeros(1, 2, 2), torch.zeros(1, 2, dtype=torch.bool))
