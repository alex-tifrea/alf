"""ICMD active learning: acquisition."""

from __future__ import annotations

import numpy as np
from alf_core.dataclasses import LabelledCandidates
from alf_core.optimizer.acquisition_function import AcquisitionFunction
from alf_core.optimizer.search import DatasetSearch
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from alf_icmd.models import catboost, embed, features, mlp_activations


def greedy_coreset(train: np.ndarray, pool: np.ndarray, batch: int) -> np.ndarray:
    """Select a batch by farthest-first traversal, updating distance after each pick."""
    distances = cdist(pool, train, metric="sqeuclidean").min(axis=1)
    selected = []
    for _ in range(batch):
        index = int(np.argmax(distances))
        selected.append(index)
        distances = np.minimum(distances, np.sum((pool - pool[index]) ** 2, axis=1))
        distances[selected] = -np.inf
    return np.asarray(selected)


def greedy_pv(covariance: np.ndarray, batch: int, noise: float) -> np.ndarray:
    """Greedily condition the candidate posterior after every pending acquisition."""
    covariance = covariance.copy()
    selected = []
    for _ in range(batch):
        scores = np.diag(covariance).copy()
        scores[selected] = -np.inf
        index = int(np.argmax(scores))
        selected.append(index)
        column = covariance[:, index].copy()
        covariance -= np.outer(column, column) / (covariance[index, index] + noise)
    return np.asarray(selected)


def activation_covariance(train: np.ndarray, pool: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Cosine activation kernel posterior from equation (8) of the PV paper."""
    train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    pool = pool / np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
    cross = pool @ train.T
    factor = cho_factor(train @ train.T + ridge * np.eye(len(train)))
    return pool @ pool.T - cross @ cho_solve(factor, cross.T)


class Acquisition(AcquisitionFunction):
    """Score only inputs, acquired labels, and the designated validation subsets."""

    def __init__(self, method: str, seed: int, scaler: StandardScaler):
        self.method, self.seed, self.scaler = method, seed, scaler
        self.network = None
        self.diagnostics = []

    def __call__(self, search_candidates, state):
        train = state.dataset.train_dataset
        validation = state.dataset.validation_dataset
        xt, xp = features(train.candidates), features(search_candidates)
        x, p = self.scaler.transform(xt), self.scaler.transform(xp)
        y = train.labels
        batch = state.acq_batch_size
        rng = np.random.default_rng(self.seed + 1009 * state.round)
        # Randomize tie order independently of the source row order.
        order = rng.permutation(len(p))
        xpo, po = xp[order], p[order]
        selected = None
        diagnostic = {"round": state.round + 1, "n_train": len(y)}
        if self.method == "uniform":
            selected = np.arange(batch)
        elif self.method == "coreset":
            selected = greedy_coreset(x, po, batch)
        elif self.method == "bootstrap":
            predictions = []
            iterations = state.surrogate.model.regressor.tree_count_
            for member in range(8):
                ids = rng.integers(0, len(y), len(y))
                model = catboost(self.seed + member + 1, iterations=iterations)
                model.fit(xt[ids], y[ids])
                predictions.append(model.predict(xpo))
            scores = np.std(predictions, axis=0, ddof=1)
        elif self.method in {"gp_std", "gp_pv"}:
            kernel = ConstantKernel(1, (0.01, 100)) * Matern(
                np.ones(x.shape[1]),
                (0.1, 20),
                nu=2.5,
            ) + WhiteKernel(0.05, (1e-4, 1))
            model = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, random_state=self.seed
            )
            model.fit(x, y)
            noise = model.kernel_.k2.noise_level
            latent = model.kernel_.k1
            cross = latent(po, x)
            posterior = latent(po) - cross @ cho_solve((model.L_, True), cross.T)
            diagnostic.update({"gp_kernel": str(model.kernel_), "gp_noise": noise})
            if self.method == "gp_pv":
                selected = greedy_pv(posterior, batch, noise)
            else:
                scores = np.maximum(np.diag(posterior), 0)
        elif self.method == "conformal_width":
            calibration = state.dataset.splits["calibration"]
            xc = features(calibration.candidates)
            models = []
            for quantile in [0.05, 0.95]:
                model = catboost(
                    self.seed, iterations=300, loss_function=f"Quantile:alpha={quantile}"
                )
                model.fit(xt, y)
                models.append(model)
            cal = np.sort(np.column_stack([model.predict(xc) for model in models]), axis=1)
            pred = np.sort(np.column_stack([model.predict(xpo) for model in models]), axis=1)
            residuals = np.maximum(cal[:, 0] - calibration.labels, calibration.labels - cal[:, 1])
            rank = int(np.ceil((len(residuals) + 1) * 0.9))
            q = max(0.0, float(np.sort(residuals)[rank - 1]))
            scores = pred[:, 1] - pred[:, 0] + 2 * q
            diagnostic.update({"cqr_q_mpa": q, "mean_pool_width_mpa": float(scores.mean())})
        else:
            refit = self.method == "activation_pv"
            if self.network is None or refit:
                xv = self.scaler.transform(features(validation.candidates))
                self.network = mlp_activations(
                    x,
                    y,
                    xv,
                    validation.labels,
                    self.seed,
                    train=self.method != "activation_random_pv",
                )
            posterior = activation_covariance(embed(self.network, x), embed(self.network, po))
            selected = greedy_pv(posterior, batch, 1e-3)
        if selected is None:
            selected = np.argsort(-scores, kind="stable")[:batch]
        selected = order[selected]
        assert len(set(selected)) == batch
        self.diagnostics.append(diagnostic)
        ranks = np.full(len(p), -1.0)
        ranks[selected] = np.arange(batch, 0, -1)
        return LabelledCandidates(search_candidates, ranks)


class MaskedDatasetSearch(DatasetSearch):
    """Omit BO regret metrics, which would unnecessarily inspect hidden pool labels."""

    def get_metrics(self, state):
        return {}
