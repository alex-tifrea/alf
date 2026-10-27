"""ICMD active learning: runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from alf_core.optimizer.optimizer import Optimizer
from alf_core.oracle.oracle import Oracle
from alf_core.surrogate.surrogate import Surrogate
from alf_core.tasks.design_task import DesignTask
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from alf_icmd.acquisition import Acquisition, MaskedDatasetSearch
from alf_icmd.constants import ALF_REVISION, FEATURES, METHODS, NAMES, PRIMARY
from alf_icmd.data import SKEWS, ICMDataset, seed_eligibility
from alf_icmd.models import CommonRegressor, features


class PredictionOnlyTask(DesignTask):
    """Save test-input predictions during AL; score test labels after all runs finish."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.records, self.predictions = [], []

    def evaluate(self, state):
        train = state.dataset.train_dataset
        validation = state.dataset.validation_dataset
        prediction = state.surrogate.predict(validation.candidates).means
        self.records.append({
            "round": state.round,
            "n_train": len(train),
            "val_rmse": root_mean_squared_error(validation.labels, prediction),
            "iterations": state.surrogate.model.regressor.tree_count_,
            "train_ids": [c.data for c in train.candidates],
        })
        self.predictions.append(
            state.surrogate.predict(state.dataset.test_dataset.candidates).means
        )
        state.round_metrics.metrics["dataset/num_train"] = len(train)
        return state


def run_one(job):
    """Execute one complete paired campaign and retain its acquisition audit trail."""
    source, output, method, seed, initial, rounds, batch, skew = job
    torch.set_num_threads(1)
    threadpool_limits(1)
    directory = Path(output) / "runs" / f"{method}_n{initial}_s{seed}"
    if (directory / "complete.json").exists():
        return str(directory.name) + " already complete"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frame = pd.read_parquet(source)
    frame = frame.loc[frame[PRIMARY].notna()].reset_index(drop=True)
    dataset = ICMDataset(frame, seed, initial, seed_skew=skew)
    dataset.setup()
    scaler = StandardScaler().fit(
        features(dataset.train_dataset.candidates + dataset.candidate_pool.candidates)
    )
    acquisition = Acquisition(method, seed, scaler)
    task = PredictionOnlyTask(acq_batch_size=batch, num_acq_rounds=rounds)
    state = task.setup(dataset, Surrogate(CommonRegressor(seed)))
    task.run(state, [], Optimizer(acquisition, MaskedDatasetSearch()), Oracle(dataset))
    for record in task.records:
        record.update({"method": method, "seed": seed, "initial": initial})
    (directory / "history.json").write_text(json.dumps(task.records, indent=2))
    (directory / "acquisition_diagnostics.json").write_text(
        json.dumps(acquisition.diagnostics, indent=2)
    )
    np.savez_compressed(
        directory / "test_predictions.npz",
        predictions=np.asarray(task.predictions),
        query_ids=np.asarray([c.data for c in dataset.test_dataset.candidates]),
    )
    elapsed = time.time() - started
    (directory / "complete.json").write_text(json.dumps({"elapsed_seconds": elapsed}))
    return f"{directory.name}: {elapsed:.1f}s"


def prepare(source: Path, output: Path, args):
    """Freeze the protocol and save dataset identities before the benchmark starts."""
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source)
    valid = frame.loc[frame[PRIMARY].notna()].reset_index(drop=True)
    assert not valid.duplicated(FEATURES).any()
    assert valid.query_id.is_unique
    assert np.isfinite(valid[FEATURES].to_numpy()).all()
    protocol = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "alf_revision": ALF_REVISION,
        "target": PRIMARY,
        "features": FEATURES + ["log_aging_duration_h"],
        "initial_sizes": args.initial_sizes,
        "seeds": args.seeds,
        "methods": args.methods,
        "rounds": args.rounds,
        "batch": args.batch,
        "split_counts": valid.split.value_counts().to_dict(),
        "validation": 52,
        "calibration": 53,
        "main_regressor": "CatBoost depth=5, lr=.05, l2=3, max600, patience60; refit each round",
        "bootstrap_members": 8,
        "activation_mlp": "7-64-64-1 ReLU; AdamW .003; max600, patience60",
        "activation_kernel": "cosine-normalized concatenated hidden activations; ridge .001",
        "gp": "ARD Matern 5/2 + white noise; marginal likelihood fit each round; no restarts",
        "conformal": "90% CQR; two 300-tree quantile models; fixed53 calibration reused; q clipped>=0",
        "test_policy": "test labels masked throughout DesignTask; score saved predictions only after campaign",
        "input_scaling": "fixed on all697 eligible training-pool inputs, without labels",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if args.seed_skew != "iid":
        _, region = seed_eligibility(valid, args.seed_skew)
        protocol["seed_region"] = region
    lock = output / "protocol.json"
    if lock.exists():
        assert json.loads(lock.read_text()) == protocol, (
            "Use a new output directory for a changed protocol."
        )
    lock.write_text(json.dumps(protocol, indent=2))
    manifest = valid[["query_id", "split", *FEATURES]].copy()
    validation = np.flatnonzero(valid.split.eq("val"))
    permutation = np.random.default_rng(20261027).permutation(validation)
    manifest.loc[permutation[:52], "role"] = "validation"
    manifest.loc[permutation[52:], "role"] = "calibration"
    manifest["role"] = manifest.role.fillna(manifest.split)
    manifest.to_csv(output / "split_manifest.csv", index=False)


def evaluate(source: Path, output: Path):
    """Unseal test labels once all prespecified acquisition campaigns have completed."""
    protocol = json.loads((output / "protocol.json").read_text())
    expected = len(protocol["methods"]) * len(protocol["seeds"]) * len(protocol["initial_sizes"])
    directories = sorted((output / "runs").glob("*/complete.json"))
    assert len(directories) == expected, (len(directories), expected)
    frame = pd.read_parquet(source).set_index("query_id")
    rows, predictions = [], []
    for complete in directories:
        directory = complete.parent
        history = json.loads((directory / "history.json").read_text())
        saved = np.load(directory / "test_predictions.npz")
        ids = saved["query_ids"].tolist()
        assert frame.loc[ids, "split"].eq("test").all()
        y = frame.loc[ids, PRIMARY].to_numpy()
        last_ids = set()
        for record, prediction in zip(history, saved["predictions"], strict=True):
            train_ids = set(record.pop("train_ids"))
            assert len(train_ids) == record["n_train"]
            assert frame.loc[list(train_ids), "split"].eq("train").all()
            assert last_ids <= train_ids
            if last_ids:
                assert len(train_ids - last_ids) == protocol["batch"]
            last_ids = train_ids
            record.update({
                "test_rmse": root_mean_squared_error(y, prediction),
                "test_mae": mean_absolute_error(y, prediction),
                "test_r2": r2_score(y, prediction),
            })
            rows.append(record)
            predictions.extend(
                {
                    "method": record["method"],
                    "initial": record["initial"],
                    "seed": record["seed"],
                    "round": record["round"],
                    "query_id": qid,
                    "prediction": float(p),
                    "target": float(target),
                }
                for qid, p, target in zip(ids, prediction, y, strict=True)
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(predictions).to_parquet(output / "predictions.parquet", index=False)
    grouped = metrics.groupby(["initial", "method", "round", "n_train"])
    summary = grouped.test_rmse.agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(output / "learning_curves.csv", index=False)
    uniform = metrics.loc[metrics.method.eq("uniform"), ["initial", "seed", "round", "test_rmse"]]
    paired = metrics.merge(uniform, on=["initial", "seed", "round"], suffixes=("", "_uniform"))
    paired["gain_percent"] = 100 * (1 - paired.test_rmse / paired.test_rmse_uniform)
    paired.to_csv(output / "paired_metrics.csv", index=False)
    return metrics, summary, paired


def plot(output: Path, summary: pd.DataFrame, paired: pd.DataFrame):
    """Render Figure-4a-style curves and paired gains at the same label budgets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    sizes = sorted(summary.initial.unique())
    colors = dict(
        zip(
            METHODS,
            [
                "#222222",
                "#0072B2",
                "#D55E00",
                "#009E73",
                "#56B4E9",
                "#CC79A7",
                "#E69F00",
                "#7B4AB5",
                "#888888",
            ],
        )
    )
    for kind in ["rmse", "gain"]:
        fig, axes = plt.subplots(1, len(sizes), figsize=(17, 5.7), sharey=True, squeeze=False)
        for ax, initial in zip(axes[0], sizes, strict=True):
            for method in METHODS:
                if kind == "rmse":
                    d = summary.loc[summary.initial.eq(initial) & summary.method.eq(method)]
                    mean, std = d["mean"].to_numpy(), d["std"].to_numpy()
                    x = d["round"].to_numpy()
                else:
                    if method == "uniform":
                        continue
                    d = paired.loc[paired.initial.eq(initial) & paired.method.eq(method)]
                    d = d.groupby("round").gain_percent.agg(["mean", "std"]).reset_index()
                    mean, std, x = d["mean"].to_numpy(), d["std"].to_numpy(), d["round"].to_numpy()
                ax.plot(
                    x,
                    mean,
                    color=colors[method],
                    label=NAMES[method],
                    marker="o",
                    ms=3,
                    lw=2.3 if method == "uniform" else 1.6,
                    linestyle="--" if method == "activation_random_pv" else "-",
                )
                ax.fill_between(x, mean - std, mean + std, color=colors[method], alpha=0.09)
            ax.set_title(f"Initial labeled set: {initial}")
            ax.set_xlabel("Acquisition round (+50 labels)")
            ax.grid(alpha=0.18)
            secondary = ax.secondary_xaxis(
                "top",
                functions=(lambda x, n=initial: n + 50 * x, lambda x, n=initial: (x - n) / 50),
            )
            secondary.set_xlabel("Total training labels")
            secondary.set_xticks(initial + 50 * np.arange(0, summary["round"].max() + 1, 2))
            if kind == "gain":
                ax.axhline(0, color="#222222", lw=1)
        axes[0, 0].set_ylabel(
            "Test RMSE (MPa) ↓" if kind == "rmse" else "Paired RMSE improvement over uniform (%) ↑"
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
        protocol = json.loads((output / "protocol.json").read_text())
        skew = protocol.get("seed_region", {}).get("skew", "iid")
        title = SKEWS.get(skew, "IID seeds")
        fig.suptitle(f"ICMD yield strength · {title} · five paired seeds · mean ±1 SD", y=1.02)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(output / f"learning_{kind}.png", dpi=180, bbox_inches="tight")
        fig.savefig(output / f"learning_{kind}.pdf", bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--initial-sizes", nargs="+", type=int, default=[25, 50, 100])
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[20261027, 20261028, 20261029, 20261030, 20261031]
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--seed-skew", choices=["iid", *SKEWS], default="iid")
    args = parser.parse_args()
    if not args.report_only:
        prepare(args.source, args.output, args)
        jobs = [
            (
                str(args.source),
                str(args.output),
                method,
                seed,
                initial,
                args.rounds,
                args.batch,
                args.seed_skew,
            )
            for initial in args.initial_sizes
            for seed in args.seeds
            for method in args.methods
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(run_one, jobs):
                print(result, flush=True)
    metrics, summary, paired = evaluate(args.source, args.output)
    plot(args.output, summary, paired)
    print(
        metrics
        .loc[metrics["round"].eq(metrics["round"].max())]
        .groupby(["initial", "method"])
        .test_rmse.mean()
    )


if __name__ == "__main__":
    main()
