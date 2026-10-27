"""ICMD active learning: data."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from alf_core.dataclasses import Candidate, LabelledCandidates
from alf_core.dataset.base_dataset import BaseDataset, BaseDatasetConfig

from alf_icmd.constants import FEATURES, PRIMARY

SKEWS = {
    "low_temperature": "Low aging temperature",
    "high_temperature": "High aging temperature",
    "lean_alloy": "Low total alloy content",
    "compact_region": "Compact input region",
}


def seed_eligibility(frame: pd.DataFrame, skew: str) -> tuple[np.ndarray, dict]:
    """Define an input-only seed region from the eligible training pool.

    Args:
        frame: Original split assignments and scalar input columns.
        skew: IID or one of the four predefined feature skews.

    Returns:
        Eligible row positions and the feature threshold defining their region.
    """
    indices = np.flatnonzero(frame.split.eq("train"))
    x = frame.iloc[indices][FEATURES].to_numpy(dtype=float)
    if skew == "iid":
        return indices, {"skew": skew, "eligible": len(indices)}
    if skew == "low_temperature":
        score = x[:, 4]
    elif skew == "high_temperature":
        score = -x[:, 4]
    elif skew == "lean_alloy":
        score = x[:, :3].sum(axis=1)
    elif skew == "compact_region":
        from sklearn.preprocessing import StandardScaler

        z = StandardScaler().fit_transform(np.column_stack([x, np.log(x[:, 5])]))
        center = int(np.argmin(np.sum((z - np.median(z, axis=0)) ** 2, axis=1)))
        score = np.sum((z - z[center]) ** 2, axis=1)
    else:
        raise ValueError(f"Unknown seed skew: {skew}")
    threshold = float(np.quantile(score, 0.35))
    eligible = indices[score <= threshold]
    metadata = {
        "skew": skew,
        "quantile": 0.35,
        "threshold": threshold,
        "eligible": len(eligible),
        "fraction": len(eligible) / len(indices),
        "includes_threshold_ties": True,
    }
    if skew == "compact_region":
        metadata["center_inputs"] = dict(zip(FEATURES, x[center].tolist(), strict=True))
    return eligible, metadata


class ICMDataset(BaseDataset):
    """Preserve source splits; expose pool and test inputs with masked labels."""

    def __init__(self, frame: pd.DataFrame, seed: int, initial: int, seed_skew: str = "iid"):
        super().__init__(
            BaseDatasetConfig(
                name="icmd_yield",
                modality="tabular",
                seed=seed,
                train_ratio=0.7,
                validation_frac=0,
                test_ratio=0.2,
                problem_type="regression",
            )
        )
        self.frame = frame
        self.initial = initial
        self.seed_skew = seed_skew

    def load_dataset(self):
        return LabelledCandidates(
            [
                Candidate(str(row.query_id), "tabular", {"x": row[FEATURES].to_list()})
                for _, row in self.frame.iterrows()
            ],
            self.frame[PRIMARY].to_numpy().copy(),
        )

    def _split_dataset(self):
        source = self.raw_dataset
        train = np.flatnonzero(self.frame.split.eq("train"))
        rng = np.random.default_rng(self.config.seed)
        eligible, self.seed_region = seed_eligibility(self.frame, self.seed_skew)
        assert self.initial <= len(eligible), "Initial size exceeds the predefined seed region."
        seed_order = rng.permutation(eligible)
        outside = np.setdiff1d(train, eligible)
        permutation = np.concatenate([seed_order, rng.permutation(outside)])
        validation = np.flatnonzero(self.frame.split.eq("val"))
        validation = np.random.default_rng(20261027).permutation(validation)
        test = np.flatnonzero(self.frame.split.eq("test"))
        indices = {
            "train": permutation[: self.initial],
            "candidate_pool": permutation[self.initial :],
            "validation": validation[:52],
            "calibration": validation[52:],
            "test": test,
        }
        splits = {key: LabelledCandidates(*source[idx]) for key, idx in indices.items()}
        for key in ["candidate_pool", "test"]:
            splits[key].labels[:] = np.nan
        self.init_candidate_pool = copy.deepcopy(splits["candidate_pool"])
        return splits

    def query(self, candidates):
        allowed = {c.data for c in self.candidate_pool.candidates}
        assert all(c.data in allowed for c in candidates)
        return super().query(candidates)

    def get_metrics(self):
        return {f"num_{key}": len(value) for key, value in self.splits.items()}
