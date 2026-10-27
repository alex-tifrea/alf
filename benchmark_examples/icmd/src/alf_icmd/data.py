"""ICMD active learning: data."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from alf_core.dataclasses import Candidate, LabelledCandidates
from alf_core.dataset.base_dataset import BaseDataset, BaseDatasetConfig

from alf_icmd.constants import FEATURES, PRIMARY


class ICMDataset(BaseDataset):
    """Preserve source splits; expose pool and test inputs with masked labels."""

    def __init__(self, frame: pd.DataFrame, seed: int, initial: int):
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
        permutation = np.random.default_rng(self.config.seed).permutation(train)
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
