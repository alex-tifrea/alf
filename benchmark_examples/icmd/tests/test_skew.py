"""Feature-only eligibility and unchanged held-out splits under seed skew."""

import numpy as np
import pandas as pd
import pytest
from alf_icmd.constants import FEATURES, PRIMARY
from alf_icmd.data import SKEWS, ICMDataset, seed_eligibility


@pytest.mark.parametrize("skew", SKEWS)
def test_skew_preserves_pool_and_nests_seeds_without_labels(skew):
    """Only initialization changes; every remaining training point stays acquirable."""
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.uniform(1, 10, (825, 6)), columns=FEATURES)
    frame["split"] = ["train"] * 700 + ["val"] * 105 + ["test"] * 20
    frame["query_id"] = [f"q{i}" for i in range(len(frame))]
    frame[PRIMARY] = rng.normal(size=len(frame))
    eligible, region = seed_eligibility(frame.drop(columns=PRIMARY), skew)
    assert 100 <= len(eligible) < 700
    assert region["skew"] == skew
    small, large = ICMDataset(frame, 3, 25, skew), ICMDataset(frame, 3, 100, skew)
    iid = ICMDataset(frame, 3, 100)
    for dataset in [small, large, iid]:
        dataset.setup()
    first = [c.data for c in small.train_dataset.candidates]
    full = [c.data for c in large.train_dataset.candidates]
    assert first == full[:25]
    assert set(full) <= set(frame.iloc[eligible].query_id)
    assert set(full + [c.data for c in large.candidate_pool.candidates]) == set(
        frame.iloc[:700].query_id
    )
    for split in ["test", "validation", "calibration"]:
        assert large.splits[split].candidates == iid.splits[split].candidates
    shuffled = frame.copy()
    shuffled[PRIMARY] = rng.permutation(shuffled[PRIMARY])
    other, _ = seed_eligibility(shuffled, skew)
    np.testing.assert_array_equal(eligible, other)


def test_iid_order_is_unchanged():
    """Adding skew support preserves the original IID permutation exactly."""
    frame = pd.DataFrame(np.ones((825, 6)), columns=FEATURES)
    frame["split"] = ["train"] * 700 + ["val"] * 105 + ["test"] * 20
    frame["query_id"] = [str(i) for i in range(len(frame))]
    frame[PRIMARY] = np.arange(len(frame), dtype=float)
    dataset = ICMDataset(frame, 3, 100)
    dataset.setup()
    order = [
        int(c.data) for c in dataset.train_dataset.candidates + dataset.candidate_pool.candidates
    ]
    np.testing.assert_array_equal(order, np.random.default_rng(3).permutation(700))
