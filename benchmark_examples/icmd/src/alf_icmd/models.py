"""ICMD active learning: models."""

from __future__ import annotations

import copy

import numpy as np
import torch
from alf_core.dataclasses import Candidate, Predictions
from alf_core.model.base_model import BaseModel
from catboost import CatBoostRegressor


def features(candidates: list[Candidate]) -> np.ndarray:
    """Extract physical inputs and log duration without accessing any labels."""
    x = np.asarray([c.features["x"] for c in candidates], dtype=float)
    return np.column_stack([x, np.log(x[:, 5])])


def catboost(seed: int, iterations: int = 600, **kwargs) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=seed,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
        **kwargs,
    )


class CommonRegressor(BaseModel):
    """Use the identical fresh CatBoost estimator for every acquisition strategy."""

    def __init__(self, seed: int):
        self.seed = seed

    def featurise(self, inputs):
        return features(inputs)

    def train(self, train_data, val_data=None):
        self.regressor = catboost(self.seed)
        self.regressor.fit(
            features(train_data.candidates),
            train_data.labels,
            eval_set=(features(val_data.candidates), val_data.labels),
            early_stopping_rounds=60,
            use_best_model=True,
        )

    def predict(self, candidate_points):
        return Predictions(means=self.regressor.predict(features(candidate_points)))

    def sample(self, condition=None):
        raise NotImplementedError("This benchmark uses a fixed offline pool.")


def mlp_activations(x, y, xv, yv, seed, train=True):
    """Fit a small tabular network and return its two concatenated hidden layers."""
    torch.manual_seed(seed)
    network = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    if train:
        center, scale = y.mean(), max(y.std(), 1e-8)
        xt = torch.tensor(x, dtype=torch.float32)
        yt = torch.tensor((y - center) / scale, dtype=torch.float32)[:, None]
        xvt = torch.tensor(xv, dtype=torch.float32)
        yvt = torch.tensor((yv - center) / scale, dtype=torch.float32)[:, None]
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.003, weight_decay=0.01)
        best, stale, weights = np.inf, 0, copy.deepcopy(network.state_dict())
        for _ in range(600):
            optimizer.zero_grad()
            loss = torch.mean((network(xt) - yt) ** 2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                value = float(torch.mean((network(xvt) - yvt) ** 2))
            if value < best - 1e-6:
                best, stale, weights = value, 0, copy.deepcopy(network.state_dict())
            else:
                stale += 1
            if stale >= 60:
                break
        network.load_state_dict(weights)
    return network


def embed(network, x):
    with torch.no_grad():
        hidden = network[:2](torch.tensor(x, dtype=torch.float32))
        second = network[2:4](hidden)
        return torch.cat([hidden, second], dim=1).numpy().astype(float)
