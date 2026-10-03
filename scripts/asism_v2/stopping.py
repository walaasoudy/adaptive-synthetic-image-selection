"""Contract §10: progressive within-class selection with adaptive stopping (WHICH and HOW MANY).

Locked methodology (Walaa, 2026-10-03), implemented here and nowhere else:
  - candidates are ranked within each class by the lower 95% bound of their own marginal utility
    (amendment of 2026-10-03, below; before it, by the point model's score);
  - synthetic images are added one at a time, each class offering its next-ranked image;
  - a class keeps adding while the lower 95% confidence bound of the estimated marginal utility of
    its next image is above 0, and stops for good when it reaches 0 or below (or runs out);
  - classes may stop at different counts; every step and every stop is recorded;
  - classifier_val and final_eval_heldout are never read: the inputs are the candidate signal
    table and a fitted ranker, nothing else.

Marginal utility comes from the fitted set-utility model itself (pipeline.py),
    U(S) = b + lam * log(1 + sum_{x in S} w(x)),  U(empty) = b,
so adding x to S is worth lam * [log(1 + W + w(x)) - log(1 + W)], W the weight already selected. Its
sign is the sign of lam * w(x): an image is judged on its own weight, not against the set's mean.
The lower bound is the 5th percentile of that difference across the bootstrap ensemble of fit_ranker
(a one-sided 95% bound). A model whose size term was frozen (all supervision at one size), or whose
class term was frozen (one class allocation), cannot estimate the value of adding an image and is
refused.

Amendment (Walaa, 2026-10-03, before any utility measurement; design check §12). Within a class the
order is by the image's own lower bound: the 5th percentile across the bootstrap models of lam * w(x),
the quantity whose sign is the sign of the image's marginal utility in any set. Highest first, ties by
image_id. The order and the stop then use the same criterion, so a class stops at the first image that
is not shown to help. Before the amendment the order was the point model's lam * w(x); the images
ranked first there have the most extreme signals and the widest bootstrap spread, so a class met its
least certain images first and could stop at one of them with hundreds of clearly useful images
behind it (dry run on planted data, design check §11). The stopping condition itself is unchanged.

At each step the image added is the offered one with the highest lower bound (ties: class name), so
the order of addition is itself a function of the learned utility. E4's q* is not read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .features import validate_frame
from .pipeline import FittedRanking

LOWER_BOUND_QUANTILE = 0.05      # one-sided 95% lower confidence bound
MIN_BOOTSTRAP_MODELS = 20
OFFER_ORDER = "lower_bound"      # within a class: by the image's own lower bound, not the point score


def _utility(lam: float, total: float) -> float:
    return lam * float(np.log1p(max(total, 0.0)))


def progressive_select(fitted: FittedRanking, candidates: pd.DataFrame) -> dict:
    """-> {"selected": DataFrame, "counts": {class: n}, "trajectory": [...], "stops": {...}}."""
    validate_frame(candidates)
    if "dx" not in candidates or candidates.dx.isna().any():
        raise ValueError("Missing diagnosis")
    if "log_count" in fitted.history.get("frozen_unidentifiable", []):
        raise ValueError("The size term was frozen (one subset size): the marginal utility of adding "
                         "an image is not identifiable, so contract §10 stopping cannot be applied")
    if "class_mix" in fitted.history.get("frozen_unidentifiable", []):
        # Progressive selection builds sets whose class fractions change at every step. With one
        # class allocation in the supervision, only one weighted average of the class terms is
        # estimated, so where each class's weights cross 0 is not estimated by anything: the
        # per-class counts would be an artifact.
        raise ValueError("The class-mix term was frozen (one class allocation): utility across "
                         "classes is not identifiable, so contract §10 stopping cannot be applied")
    if len(fitted.ensemble) < MIN_BOOTSTRAP_MODELS:
        raise ValueError(f"A lower confidence bound needs >= {MIN_BOOTSTRAP_MODELS} bootstrap models; "
                         f"the ranker has {len(fitted.ensemble)}")
    classes = fitted.classes
    unknown = set(candidates.dx.astype(str)) - set(classes)
    if unknown:
        raise ValueError(f"Unknown diagnosis {sorted(unknown)}")

    x = torch.tensor(fitted.normalizer.transform(candidates), dtype=torch.float32)
    cls = torch.tensor([classes.index(str(d)) for d in candidates.dx], dtype=torch.long)
    models = [fitted.model, *fitted.ensemble]
    with torch.no_grad():
        scores = fitted.model.score(x, cls).numpy().astype(float)
        weights = np.stack([m.image_weight(x, cls).numpy().astype(float) for m in models])   # [1+B, n]
        lams = np.array([float(m.log_count) for m in models])
    if not np.isfinite(weights).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite ranking score")

    ids = candidates.image_id.astype(str).to_numpy()
    own_lower = np.quantile(lams[1:, None] * weights[1:], LOWER_BOUND_QUANTILE, axis=0)
    order = {}
    for c, name in enumerate(classes):
        members = np.flatnonzero(cls.numpy() == c)
        # rank within class by the image's own lower bound of lam * w; ties by image_id
        order[name] = sorted(members, key=lambda i: (-own_lower[i], ids[i]))

    n, totals = 0, np.zeros(len(models))
    position = {name: 0 for name in classes}
    active = {name for name in classes if order[name]}
    stops = {name: {"reason": "no candidates", "at_count": 0} for name in classes if not order[name]}
    trajectory, chosen = [], []

    def offer(name):
        i = order[name][position[name]]
        gains = [_utility(lams[m], totals[m] + weights[m, i]) - _utility(lams[m], totals[m])
                 for m in range(len(models))]
        return i, gains[0], float(np.quantile(gains[1:], LOWER_BOUND_QUANTILE))

    while active:
        offers = {name: offer(name) for name in sorted(active)}
        for name, (i, point, lower) in offers.items():
            if lower <= 0:
                stops[name] = {"reason": "lower 95% bound of marginal utility <= 0",
                               "at_count": position[name], "next_image_id": ids[i],
                               "marginal_point": point, "marginal_lower_bound": lower}
                active.discard(name)
        if not active:
            break
        name = max(sorted(active), key=lambda k: offers[k][2])
        i, point, lower = offers[name]
        totals += weights[:, i]
        n += 1
        position[name] += 1
        chosen.append(i)
        trajectory.append({"step": n, "dx": name, "image_id": ids[i], "rank_in_class": position[name] - 1,
                           "ranking_score": float(scores[i]), "image_weight": float(weights[0, i]),
                           "own_lower_bound": float(own_lower[i]),
                           "marginal_point": point,
                           "marginal_lower_bound": lower, "set_size": n})
        if position[name] == len(order[name]):
            stops[name] = {"reason": "class exhausted", "at_count": position[name]}
            active.discard(name)

    selected = candidates.iloc[chosen][["image_id", "dx"]].assign(
        ranking_score=scores[chosen], image_weight=weights[0, chosen],
        score_source=fitted.score_source(ids[chosen])).reset_index(drop=True)
    return {"selected": selected,
            "counts": {name: position[name] for name in classes},
            "trajectory": trajectory, "stops": stops,
            "rule": "contract §10: add while the one-sided 95% lower bound of marginal utility > 0",
            "offer_order": OFFER_ORDER,
            "lower_bound_quantile": LOWER_BOUND_QUANTILE, "bootstrap_models": len(fitted.ensemble)}
