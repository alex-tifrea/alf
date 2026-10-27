"""Fixed-split and label-isolation checks."""

import numpy as np
import pandas as pd
from alf_icmd.constants import FEATURES, PRIMARY
from alf_icmd.data import ICMDataset


def test_splits_mask_pool_and_test_and_keep_seeds_nested():
    rng = np.random.default_rng(10)
    frame = pd.DataFrame(rng.uniform(1, 10, size=(220, 6)), columns=FEATURES)
    frame["query_id"] = [f"q{i}" for i in range(220)]
    frame[PRIMARY] = rng.normal(size=220)
    frame["split"] = ["train"] * 100 + ["val"] * 105 + ["test"] * 15
    small, large = ICMDataset(frame, 10, 25), ICMDataset(frame, 10, 50)
    small.setup()
    large.setup()
    assert [c.data for c in small.train_dataset.candidates] == [
        c.data for c in large.train_dataset.candidates[:25]
    ]
    assert np.isnan(small.candidate_pool.labels).all()
    assert np.isnan(small.test_dataset.labels).all()
    assert len(small.validation_dataset) == 52
    assert len(small.splits["calibration"]) == 53
    validation_ids = [c.data for c in small.validation_dataset.candidates]
    candidates = small.candidate_pool.candidates[:10]
    labelled = small.query(candidates)
    assert np.isfinite(labelled.labels).all()
    small.update_splits(labelled)
    assert len(small.train_dataset) == 35
    assert len(small.candidate_pool) == 65
    assert validation_ids == [c.data for c in small.validation_dataset.candidates]


def test_hidden_labels_do_not_change_splits_or_input_kernel():
    rng = np.random.default_rng(10)
    frame = pd.DataFrame(rng.uniform(1, 10, size=(220, 6)), columns=FEATURES)
    frame["query_id"] = [f"q{i}" for i in range(220)]
    frame[PRIMARY] = rng.normal(size=220)
    frame["split"] = ["train"] * 100 + ["val"] * 105 + ["test"] * 15
    first = ICMDataset(frame, 10, 25)
    first.setup()
    hidden = {c.data for c in first.candidate_pool.candidates + first.test_dataset.candidates}
    changed = frame.copy()
    changed.loc[changed.query_id.isin(hidden), PRIMARY] *= -10000
    second = ICMDataset(changed, 10, 25)
    second.setup()
    np.testing.assert_array_equal(first.train_dataset.labels, second.train_dataset.labels)
    assert first.candidate_pool.candidates == second.candidate_pool.candidates
    assert first.test_dataset.candidates == second.test_dataset.candidates
