"""Four-signal V2 input contract; fit preprocessing on training images ONLY.

No fixed beneficial direction is assigned to uncertainty or IQA. These are
model inputs, not hand-weighted utility labels. Clinical value requires ablation.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

SIGNALS = ("similarity_knn_mean", "iqa_composite",
           "uncertainty_mutual_information", "explainability_calibrated_typicality")


def validate_frame(frame: pd.DataFrame) -> None:
    required = {"image_id", *SIGNALS}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Missing four-signal features or empty frame")
    if frame.image_id.isna().any() or frame.image_id.astype(str).duplicated().any():
        raise ValueError("Missing or duplicate image IDs")
    if not np.isfinite(frame[list(SIGNALS)].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite signal: explicit upstream repair required")


@dataclass(frozen=True)
class TrainingStandardizer:
    train_ids: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, frame: pd.DataFrame, train_ids: list[str]):
        validate_frame(frame)
        if not train_ids or len(set(train_ids)) != len(train_ids):
            raise ValueError("Empty or duplicate training IDs")
        indexed = frame.set_index(frame.image_id.astype(str))
        if set(train_ids) - set(indexed.index):
            raise ValueError("Missing training image features")
        values = indexed.loc[train_ids, list(SIGNALS)].to_numpy(dtype=float)
        scale = values.std(axis=0)
        scale[scale == 0] = 1.0
        return cls(tuple(train_ids), tuple(values.mean(axis=0)), tuple(scale))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        validate_frame(frame)
        return (frame[list(SIGNALS)].to_numpy(dtype=float) - self.mean) / self.scale
