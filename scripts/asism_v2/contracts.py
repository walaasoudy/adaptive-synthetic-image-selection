"""Fail-closed contracts for V2; no historical pipeline imports or output defaults."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    """Exclusive create: even identical existing evidence must never be replaced."""
    payload = json.dumps(value, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def validate_roles(roles: dict[str, list[str]]) -> None:
    """Inputs are lesion IDs (not image IDs). Protected outcomes cannot enter V2 development."""
    allowed = {"classifier_train", "utility_train", "utility_validation", "utility_test"}
    if not roles or set(roles) - allowed:
        raise ValueError("Unknown/protected split role")
    seen: set[str] = set()
    for role, ids in roles.items():
        if not ids or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f"Empty, duplicate or overlapping lesion IDs: {role}")
        seen.update(ids)


def validate_measurements(rows: list[dict], subsets: dict[str, list[str]],
                          seeds: list[int], protocol: dict) -> dict[str, list[float]]:
    """Require the COMPLETE declared subset-by-seed grid, not the successful rows only.

    The target is ABSOLUTE augmented performance. Historical diagnostics found
    that subtracting real-only performance increased noise. Real-only metrics
    may be recorded elsewhere for interpretation, but cannot define this label.
    """
    if not subsets or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Empty grid or duplicated seeds")
    identities = set()
    for ids in subsets.values():
        key = tuple(sorted(ids))
        if not ids or len(key) != len(set(key)) or key in identities:
            raise ValueError("Empty, repeated or internally duplicated subset")
        identities.add(key)
    expected = {(key, seed) for key in subsets for seed in seeds}
    observed = {}
    for row in rows:
        key = (row["subset_id"], row["seed"])
        if key not in expected or key in observed:
            raise ValueError("Unexpected or repeated measurement cell")
        if row["protocol_sha256"] != fingerprint(protocol):
            raise ValueError("Stale protocol/configuration")
        if row["members_sha256"] != fingerprint({"ids": sorted(subsets[key[0]])}):
            raise ValueError("Stale subset membership")
        a = row["augmented_metric"]
        if not math.isfinite(a) or not 0 <= a <= 1:
            raise ValueError("Invalid metric")
        observed[key] = a
    if set(observed) != expected:
        raise ValueError("Incomplete declared measurement grid")
    return {key: [observed[key, seed] for seed in seeds] for key in subsets}


def validate_cache(metadata: dict, expected: dict, payload: Path) -> None:
    from .preserve import sha256
    if not expected or any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Missing or stale prediction provenance")
    if metadata.get("payload_sha256") != sha256(payload):
        raise ValueError("Prediction payload hash mismatch")
