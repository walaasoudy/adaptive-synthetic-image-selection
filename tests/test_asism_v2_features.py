import numpy as np
import pandas as pd
import pytest

from scripts.asism_v2.features import SIGNALS, TrainingStandardizer


def fixture():
    return pd.DataFrame({"image_id": ["train_a", "train_b", "validation"],
                         **{column: [1., 3., 100.] for column in SIGNALS}})


def test_validation_distribution_does_not_fit_preprocessing():
    frame = fixture()
    first = TrainingStandardizer.fit(frame, ["train_a", "train_b"])
    frame.loc[2, list(SIGNALS)] = -10000
    second = TrainingStandardizer.fit(frame, ["train_a", "train_b"])
    assert first == second
    assert first.mean == (2., 2., 2., 2.)


@pytest.mark.parametrize("column", SIGNALS)
def test_each_signal_reaches_its_own_feature_channel(column):
    frame = fixture()
    fitted = TrainingStandardizer.fit(frame, ["train_a", "train_b"])
    before = fitted.transform(frame)
    frame.loc[2, column] += 1
    delta = fitted.transform(frame) - before
    assert delta[2, SIGNALS.index(column)] == 1
    assert np.count_nonzero(delta) == 1


def test_missing_iqa_is_not_silently_dropped():
    with pytest.raises(ValueError):
        TrainingStandardizer.fit(fixture().drop(columns="iqa_composite"), ["train_a"])


def test_missing_member_is_not_silently_dropped():
    with pytest.raises(ValueError):
        TrainingStandardizer.fit(fixture(), ["absent"])
