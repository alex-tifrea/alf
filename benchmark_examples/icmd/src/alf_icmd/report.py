"""Audit and package the prespecified ICMD active learning experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from alf_icmd.constants import NAMES
from alf_icmd.data import ICMDataset
from alf_icmd.models import CommonRegressor


def audit(output: Path) -> dict:
    """Verify paired seeds, nested acquisition, held-out identity, and round-zero equality."""
    protocol = json.loads((output / "protocol.json").read_text())
    manifest = pd.read_csv(output / "split_manifest.csv")
    test_ids = set(manifest.loc[manifest.role.eq("test"), "query_id"])
    allowed = set(manifest.loc[manifest.role.eq("train"), "query_id"])
    histories = {}
    for path in sorted((output / "runs").glob("*/history.json")):
        history = json.loads(path.read_text())
        first = history[0]
        key = (first["initial"], first["seed"])
        ids = first["train_ids"]
        if key in histories:
            assert histories[key] == ids
        histories[key] = ids
        assert len(history) == protocol["rounds"] + 1
        previous = set()
        for record in history:
            current = set(record["train_ids"])
            assert (
                len(current)
                == record["n_train"]
                == first["initial"] + record["round"] * protocol["batch"]
            )
            assert previous <= current <= allowed
            assert current.isdisjoint(test_ids)
            previous = current
        saved = np.load(path.parent / "test_predictions.npz")
        assert set(saved["query_ids"]) == test_ids
        assert np.isfinite(saved["predictions"]).all()
    for seed in protocol["seeds"]:
        sizes = sorted(protocol["initial_sizes"])
        for smaller, larger in zip(sizes, sizes[1:]):
            assert histories[(smaller, seed)] == histories[(larger, seed)][:smaller]
    metrics = pd.read_csv(output / "metrics.csv")
    assert (
        metrics
        .loc[metrics["round"].eq(0)]
        .groupby(["initial", "seed"])
        .test_rmse.nunique()
        .eq(1)
        .all()
    )
    result = {
        "campaigns": len(list((output / "runs").glob("*/complete.json"))),
        "fitted_evaluators": len(metrics),
        "paired_seeds": True,
        "nested_seed_sizes": True,
        "round_zero_identical": True,
        "no_duplicate_acquisitions": True,
        "fixed_test_ids": len(test_ids),
        "train_only_acquisitions": True,
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2))
    return result


def full_pool_reference(source: Path, output: Path, seeds: list[int]):
    """Fit the same evaluator with all eligible training labels after acquisition is over."""
    from sklearn.metrics import root_mean_squared_error

    frame = pd.read_parquet(source)
    frame = frame.loc[frame.yield_strength_mpa.notna()].reset_index(drop=True)
    rows = []
    for seed in seeds:
        dataset = ICMDataset(frame, seed, int(frame.split.eq("train").sum()))
        dataset.setup()
        model = CommonRegressor(seed)
        model.setup(dataset)
        model.train(dataset.train_dataset, dataset.validation_dataset)
        prediction = model.predict(dataset.test_dataset.candidates).means
        target = frame.loc[frame.split.eq("test"), "yield_strength_mpa"].to_numpy()
        rows.append({
            "seed": seed,
            "n_train": len(dataset.train_dataset),
            "test_rmse": root_mean_squared_error(target, prediction),
        })
    pd.DataFrame(rows).to_csv(output / "full_pool_reference.csv", index=False)
    return pd.DataFrame(rows)


def summarize(output: Path):
    metrics = pd.read_csv(output / "metrics.csv")
    final_round = metrics["round"].max()
    rows = []
    for (initial, method, seed), frame in metrics.groupby(["initial", "method", "seed"]):
        frame = frame.sort_values("round")
        rows.append({
            "initial": initial,
            "method": method,
            "seed": seed,
            "aulc": np.trapezoid(frame.test_rmse, frame.n_train)
            / (frame.n_train.max() - frame.n_train.min()),
            "final_rmse": frame.loc[frame["round"].eq(final_round), "test_rmse"].iloc[0],
        })
    scores = pd.DataFrame(rows)
    uniform = scores.loc[scores.method.eq("uniform")]
    paired = scores.merge(uniform, on=["initial", "seed"], suffixes=("", "_uniform"))
    paired["final_gain_percent"] = 100 * (1 - paired.final_rmse / paired.final_rmse_uniform)
    paired["aulc_gain_percent"] = 100 * (1 - paired.aulc / paired.aulc_uniform)
    paired["final_difference_mpa"] = paired.final_rmse - paired.final_rmse_uniform
    paired.to_csv(output / "campaign_scores.csv", index=False)
    rows = []
    for (initial, method), frame in paired.groupby(["initial", "method"]):
        difference = frame.final_difference_mpa
        half = t.ppf(0.975, len(frame) - 1) * difference.std() / np.sqrt(len(frame))
        auc_difference = frame.aulc - frame.aulc_uniform
        auc_half = t.ppf(0.975, len(frame) - 1) * auc_difference.std() / np.sqrt(len(frame))
        rows.append({
            "initial": initial,
            "method": method,
            "final_rmse_mean": frame.final_rmse.mean(),
            "final_rmse_sd": frame.final_rmse.std(),
            "final_gain_percent": frame.final_gain_percent.mean(),
            "aulc_mean": frame.aulc.mean(),
            "aulc_gain_percent": frame.aulc_gain_percent.mean(),
            "paired_difference_mpa": difference.mean(),
            "paired_ci95_low": difference.mean() - half,
            "paired_ci95_high": difference.mean() + half,
            "aulc_difference_mpa": auc_difference.mean(),
            "aulc_ci95_low": auc_difference.mean() - auc_half,
            "aulc_ci95_high": auc_difference.mean() + auc_half,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    protocol = json.loads((output / "protocol.json").read_text())
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == protocol["source_sha256"]
    checks = audit(output)
    summary = summarize(output)
    reference = full_pool_reference(args.source, output, protocol["seeds"])
    tables = []
    for initial in sorted(summary.initial.unique()):
        frame = summary.loc[summary.initial.eq(initial)].sort_values("aulc_mean")
        display = pd.DataFrame({
            "Acquisition": frame.method.map(NAMES),
            "Final RMSE ± SD (MPa)": [
                f"{m:.2f} ± {s:.2f}" for m, s in zip(frame.final_rmse_mean, frame.final_rmse_sd)
            ],
            "Final gain vs uniform": [f"{v:+.1f}%" for v in frame.final_gain_percent],
            "Average curve RMSE": [f"{v:.2f}" for v in frame.aulc_mean],
            "Curve gain vs uniform": [f"{v:+.1f}%" for v in frame.aulc_gain_percent],
            "Paired final Δ 95% CI (MPa)": [
                f"[{lo:+.2f}, {hi:+.2f}]"
                for lo, hi in zip(frame.paired_ci95_low, frame.paired_ci95_high)
            ],
        })
        tables.append(
            f"<h2>Initial {initial} → final {initial + 500} labels</h2>"
            + display.to_html(index=False, border=0)
        )
    winners = summary.sort_values("aulc_mean").groupby("initial").first().reset_index()
    finding = "; ".join(
        f"initial {int(row.initial)}: {NAMES[row.method]} ({row.aulc_gain_percent:+.1f}% curve improvement)"
        for row in winners.itertuples()
    )
    final_nonuniform = summary.loc[summary.method.ne("uniform")]
    intervals_include_zero = (
        (final_nonuniform.paired_ci95_low <= 0) & (final_nonuniform.paired_ci95_high >= 0)
    ).sum()
    code = output / "code"
    code.mkdir(exist_ok=True)
    package = Path(__file__).parent
    if package.resolve() != (code / "alf_icmd").resolve():
        shutil.copytree(
            package,
            code / "alf_icmd",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    requirements = package.parents[1] / "requirements.txt"
    if requirements.exists():
        shutil.copy2(requirements, code / "requirements.txt")
    if args.source.resolve() != (output / "dataset.parquet").resolve():
        shutil.copy2(args.source, output / "dataset.parquet")
    (output / "code_hashes.json").write_text(
        json.dumps(
            {
                str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in code.rglob("*")
                if p.is_file()
            },
            indent=2,
        )
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>ICMD · Offline active learning</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1280px;margin:45px auto;padding:0 28px;color:#1b2737;line-height:1.55}}
h1{{font-size:36px}}h2{{margin-top:36px}}p,li{{max-width:1050px}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:9px;text-align:right;border-bottom:1px solid #dde3eb}}th:first-child,td:first-child{{text-align:left}}
.lead{{font-size:19px}}.note{{background:#eef3f8;padding:20px;border-radius:9px}}code{{font-size:12px}}a{{color:#1264a3}}
</style></head><body><p>QUESTEK / ICMD · 27 OCTOBER 2026 · ALF BENCHMARK</p>
<h1>Which acquisitions improve yield-strength prediction?</h1>
<p class="lead">Nine acquisition strategies, three initial labeled-set sizes, five paired seeds, and ten rounds of 50 labels.
All curves use the same CatBoost evaluator on the same 200 test points.</p>
<p><strong>Lowest average learning-curve error:</strong> {finding}.</p>
<p><strong>Uniform remains competitive.</strong> The best average-curve gains are below 1%.
{intervals_include_zero} of {len(final_nonuniform)} final paired 95% confidence intervals overlap zero improvement.
Learned MLP Activation-PV does not show a consistent advantage over the random-feature control.</p>
<img src="learning_rmse.png" alt="Test RMSE versus acquisition round, three initial set sizes">
<p>Shading: ±1 sample standard deviation over five paired seeds, conditional on one fixed data split.
Top axes show total acquired training labels. Lower RMSE is better. Fixed validation/calibration labels are additional shared overhead.</p>
<p><a href="learning_rmse.pdf">Vector PDF</a> · <a href="learning_rmse.png">PNG</a> ·
<a href="metrics.csv">Every round's metrics</a> · <a href="summary.csv">Summary CSV</a> ·
<a href="protocol.json">Frozen protocol</a> · <a href="split_manifest.csv">Split identities</a> ·
<a href="campaign_scores.csv">Paired campaign scores</a> · <a href="predictions.parquet">Test predictions</a></p>
<h2>Paired improvement over uniform acquisition</h2><img src="learning_gain.png" alt="Paired RMSE gains relative to uniform">
<p>Positive values mean lower error than uniform at the same budget and with the same initial seed set.
Average curve RMSE is trapezoidal area divided by the 500 acquired labels; it includes round zero.
Tables are sorted by that metric. Confidence intervals are unadjusted paired t intervals over five seeds;
they describe seed variability, do not include test-set resampling, and are not corrected for multiple comparisons.</p>
{"".join(tables)}
<p>Full-pool reference with the same evaluator and validation protocol, using all 697 training labels:
<strong>{reference.test_rmse.mean():.2f} ± {reference.test_rmse.std():.2f} MPa</strong>.
This is an empirical reference, not a mathematical lower bound.</p>
<h2>Data and leakage controls</h2><ul>
<li>Use yield strength in MPa from the linked regression table. Preserve its first-call configuration deduplication and original split assignments.
Of 1,182 unique configurations, 1,002 have usable yield labels: 697 train/pool, 105 validation, 200 test. Missing labels are excluded from the offline oracle.</li>
<li>Use Cu/Mg/Zn weight percent, solution temperature, aging temperature, aging duration, plus log aging duration.
No outputs, other properties, query IDs, or provenance fields are model inputs.</li>
<li>The original validation set is partitioned once into 52 early-stopping points and 53 calibration points.
Neither subset is acquired or fitted as training data. The same partition serves every run.</li>
<li>Seeds 20261027–20261031 determine initial-set permutations. Sizes 25/50/100 are nested per seed and identical across methods.
All methods acquire 50 unique remaining points per round. Fixed input scaling uses only eligible train/pool features.</li>
<li>Pool/test split labels are masked; the raw offline oracle retains archived labels. Acquisition implementations use only inputs, acquired labels, and their designated validation subsets. The oracle reveals selected training-pool labels only.
Test predictions are recorded during the loop and scored after all 135 campaigns complete. All strategies and hyperparameters were fixed before test scoring; no test-driven retuning was performed.</li>
</ul><h2>Models and acquisition definitions</h2><ul>
<li><strong>Common evaluator:</strong> fresh CatBoost, depth 5, learning rate .05, L2=3, up to 600 trees, early stopping patience 60 on the fixed52 validation points.
The same model seed is used across acquisition methods. This isolates the utility of acquired datasets for this evaluator; it is not a ranking of complete model-and-acquisition systems.</li>
<li><strong>Uniform:</strong> sample without replacement. <strong>CoreSet:</strong> greedy farthest-first Euclidean coverage in standardized input features, updating distances after each point.</li>
<li><strong>Bootstrap:</strong> eight independently row-resampled CatBoost models, each using the current common evaluator's selected tree count; select the largest prediction standard deviations.</li>
<li><strong>GP uncertainty:</strong> exact ARD Matérn-5/2 GP with optimized amplitude, length scales, and white noise each round, no optimizer restarts;
select top marginal latent variances. <strong>GP batch PV:</strong> the same GP, but condition on each pending input sequentially to reduce batch redundancy.</li>
<li><strong>Conformal width:</strong> 5th/95th percentile CatBoost regressors (300 trees), sorted bounds, 90% split-conformal calibration on the fixed53 points;
nonnegative additive correction using order statistic ceil((53+1)×.9). The correction is common to all candidates, so ranking equals uncalibrated quantile-width ranking.</li>
<li><strong>Activation-PV:</strong> cosine kernel of concatenated 64+64 ReLU hidden activations, kernel ridge λ=.001, greedy conditional posterior variance.
MLP architecture 7–64–64–1; standardized acquired labels, AdamW .003, weight decay .01, up to600 epochs, patience60 on fixed validation.
Compare refitting each round, training only on the initial seed then freezing, and an untrained random network. All versions condition on the expanding labeled input set.</li>
</ul><div class="note"><strong>Interpretation limits.</strong> This is a tabular adaptation of Activation-PV, not a replication of the paper's pretrained MACE results.
There is no compatible externally pretrained atomic model for these scalar inputs. Calibration is reused through an adaptive acquisition trajectory;
ordinary exchangeability-based conformal coverage is not guaranteed. CQR width is evaluated as an acquisition heuristic.
Results apply to archived, label-available ICMD configurations under this fixed split and evaluator; they do not establish performance on new alloy/process distributions or validate the simulator's physics.</div>
<h2>Framework, verification, and reproduction</h2>
<p>Upstream <a href="https://github.com/instadeepai/alf/tree/{protocol["alf_revision"]}">ALF commit {protocol["alf_revision"][:12]}</a> runs
DesignTask → Optimizer.ask → Oracle → State.update → Optimizer.tell. Custom adapters preserve fixed splits and defer test scoring.
<a href="audit.json">Audit</a>: {checks["campaigns"]} campaigns, {checks["fitted_evaluators"]} round evaluators, paired/nested seeds,
identical round-zero predictions, no repeated acquisitions, fixed test identities, and train-only acquisitions. Three focused tests passed,
including sequential PV against explicit kernel reconditioning.</p>
<p>GP optimizer bound warnings are retained in the execution log. The frozen bounds are .1–20 for scaled length scales,
.01–100 for amplitude, and .0001–1 for noise; no post-test retuning was performed.</p>
<p><a href="code/alf_icmd/runner.py">Runner</a> · <a href="code/alf_icmd/report.py">Report builder</a> ·
<a href="code/requirements.txt">Pinned requirements</a> · <a href="dataset.parquet">Dataset snapshot</a></p>
<p>Sources: <a href="https://arxiv.org/pdf/2609.31197">ALF paper</a> (Figure4a visual style; fixed-pool protocol is their offline setting),
<a href="https://arxiv.org/pdf/2605.03964">Activation-PV paper</a> (cosine kernel and greedy posterior variance), and
<a href="http://hermes.data.svc.od2.prometheus.co/oci/us-dallas-1/x-dallas/alextifrea/materials/icmd/questek_icmd_regression_2026-10-26/index.html">source ICMD report</a>.</p>
<p>Dataset SHA-256: <code>{protocol["source_sha256"]}</code></p></body></html>"""
    (output / "icmd-active-learning.html").write_text(html)
    (output / "index.html").write_text(
        '<!doctype html><html><head><title>ICMD · Active learning</title><meta http-equiv="refresh" content="0; url=icmd-active-learning.html"></head><body><a href="icmd-active-learning.html">ICMD · Active learning</a></body></html>'
    )
    (output / "README.md").write_text("""# ICMD active learning

The HTML report and PNG/PDF learning curves are in this directory. Generated with Python 3.12.
Source labels, fixed split identities, per-round acquired IDs, predictions, protocol, and summary tables are retained.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r code/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=code .venv/bin/python -m alf_icmd.runner --source dataset.parquet --output "$HOME/.reports/questek_icmd_al_reproduction" --workers 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=code .venv/bin/python -m alf_icmd.report --source dataset.parquet --output "$HOME/.reports/questek_icmd_al_reproduction"
```

To reproduce only the plots from existing metrics, use the runner's `--report-only` option.
Calibration labels and early-stopping labels are a shared fixed overhead, separate from the acquisition budget.
The conformal-width acquisition rank is the same as raw CQR width because calibration adds a common scalar.
""")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
