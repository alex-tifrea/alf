"""Compare feature-skewed initialization against the preserved IID benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from alf_icmd.constants import FEATURES, METHODS, NAMES, PRIMARY
from alf_icmd.data import SKEWS, ICMDataset, seed_eligibility
from alf_icmd.report import audit

DESCRIPTIONS = {
    "low_temperature": "Aging temperature at or below the training-pool 35th percentile.",
    "high_temperature": "Aging temperature at or above the training-pool 65th percentile.",
    "lean_alloy": "Sum of Cu, Mg, and Zn weight percentages at or below its 35th percentile.",
    "compact_region": "Closest 35% to a fixed central recipe in standardized seven-feature space; the recipe is the pool point nearest its coordinate-wise median.",
}


def validate_regions(source: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit every initial ID against its label-independent region and fixed baseline splits."""
    frame = pd.read_parquet(source)
    frame = frame.loc[frame[PRIMARY].notna()].reset_index(drop=True)
    diagnostics, shifts = [], []
    pool = frame.loc[frame.split.eq("train"), FEATURES]
    scaler = StandardScaler().fit(pool)
    for skew in SKEWS:
        indices, region = seed_eligibility(frame.drop(columns=PRIMARY), skew)
        allowed = set(frame.iloc[indices].query_id)
        audit(output / skew)
        for path in (output / skew / "runs").glob("*/history.json"):
            history = json.loads(path.read_text())
            initial = history[0]
            assert set(initial["train_ids"]) <= allowed
            if initial["method"] != "uniform":
                continue
            selected = frame.set_index("query_id").loc[initial["train_ids"]]
            for feature, mean in zip(
                FEATURES, scaler.transform(selected[FEATURES]).mean(axis=0), strict=True
            ):
                shifts.append({
                    "skew": skew,
                    "initial": initial["initial"],
                    "seed": initial["seed"],
                    "feature": feature,
                    "standardized_mean_shift": float(mean),
                })
        reference = ICMDataset(frame, 20261027, 25)
        reference.setup()
        experiment = ICMDataset(frame, 20261027, 25, skew)
        experiment.setup()
        for split in ["validation", "calibration", "test"]:
            assert reference.splits[split].candidates == experiment.splits[split].candidates
        diagnostics.append(region)
    pd.DataFrame(diagnostics).to_json(output / "seed_regions.json", orient="records", indent=2)
    shifts = pd.DataFrame(shifts)
    shifts.to_csv(output / "seed_feature_shifts.csv", index=False)
    return pd.DataFrame(diagnostics), shifts


def overview_figures(summary: pd.DataFrame, shifts: pd.DataFrame, output: Path):
    """Visualize paired acquisition gains and the observed input shift in initial sets."""
    columns = [(skew, initial) for skew in SKEWS for initial in [25, 50, 100]]
    methods = [method for method in METHODS if method != "uniform"]
    table = summary.pivot(index="method", columns=["skew", "initial"], values="aulc_gain_percent")
    values = table.loc[methods, columns].to_numpy()
    fig, ax = plt.subplots(figsize=(15, 6))
    bound = max(5, float(np.ceil(np.abs(values).max())))
    im = ax.imshow(values, aspect="auto", cmap="RdBu", vmin=-bound, vmax=bound)
    ax.set_yticks(range(len(methods)), [NAMES[m] for m in methods])
    ax.set_xticks(
        range(len(columns)), [f"{SKEWS[s]}\nn₀={n}" for s, n in columns], rotation=35, ha="right"
    )
    for i in range(len(methods)):
        for j in range(len(columns)):
            ax.text(
                j,
                i,
                f"{values[i, j]:+.1f}",
                ha="center",
                va="center",
                color="white" if abs(values[i, j]) > bound * 0.6 else "black",
                fontsize=9,
            )
    for boundary in [2.5, 5.5, 8.5]:
        ax.axvline(boundary, color="white", lw=3)
    ax.set_title(
        "Average learning-curve RMSE improvement over uniform (%) · same skew and seed", pad=14
    )
    fig.colorbar(im, ax=ax, label="Positive = better than uniform")
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"skew_acquisition_gains.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    means = (
        shifts
        .loc[shifts.initial.eq(100)]
        .groupby(["skew", "feature"])
        .standardized_mean_shift.mean()
        .unstack()
    )
    values = means.loc[list(SKEWS), FEATURES].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4))
    bound = max(1, np.abs(values).max())
    im = ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_yticks(range(4), list(SKEWS.values()))
    ax.set_xticks(
        range(6), ["Cu wt%", "Mg wt%", "Zn wt%", "Solution temp", "Aging temp", "Aging duration"]
    )
    for i in range(4):
        for j in range(6):
            ax.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center")
    ax.set_title(
        "Initial-set feature shift relative to the full training pool · n₀=100 · five-seed mean"
    )
    fig.colorbar(im, ax=ax, label="Training-pool standard deviations")
    fig.tight_layout()
    fig.savefig(output / "seed_feature_shifts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    """Write a separate descriptive report without changing the archived IID results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iid-report", type=Path, required=True)
    parser.add_argument("--iid-url", required=True)
    args = parser.parse_args()
    output = args.output
    suite = json.loads((output / "suite_protocol.json").read_text())
    baseline = json.loads((args.iid_report / "protocol.json").read_text())
    assert (
        suite["source_sha256"]
        == baseline["source_sha256"]
        == hashlib.sha256(args.source.read_bytes()).hexdigest()
    )
    for name in ["initial_sizes", "seeds", "methods", "rounds", "batch"]:
        assert suite[name] == baseline[name]
    for skew in SKEWS:
        pd.testing.assert_frame_equal(
            pd.read_csv(output / skew / "split_manifest.csv"),
            pd.read_csv(args.iid_report / "split_manifest.csv"),
        )
    regions, shifts = validate_regions(args.source, output)
    summary = pd.concat([
        pd.read_csv(output / skew / "summary.csv").assign(skew=skew) for skew in SKEWS
    ])
    metrics = pd.concat([
        pd.read_csv(output / skew / "metrics.csv").assign(skew=skew) for skew in SKEWS
    ])
    summary.to_csv(output / "all_summaries.csv", index=False)
    metrics.to_csv(output / "all_metrics.csv", index=False)
    iid = pd.read_csv(args.iid_report / "metrics.csv")
    compared = metrics.merge(
        iid[["method", "seed", "initial", "round", "test_rmse"]],
        on=["method", "seed", "initial", "round"],
        suffixes=("", "_iid"),
    )
    compared["skew_penalty_mpa"] = compared.test_rmse - compared.test_rmse_iid
    compared.to_csv(output / "paired_skew_vs_iid.csv", index=False)
    overview_figures(summary, shifts, output)
    best = (
        summary
        .sort_values("aulc_mean")
        .groupby(["skew", "initial"], sort=False)
        .first()
        .reset_index()
    )
    best_table = pd.DataFrame({
        "Seed region": best.skew.map(SKEWS),
        "Initial labels": best.initial,
        "Lowest curve error": best.method.map(NAMES),
        "Curve gain vs uniform (%)": best.aulc_gain_percent.round(2),
        "Paired curve Δ CI95 (MPa)": [
            f"[{lo:+.2f}, {hi:+.2f}]" for lo, hi in zip(best.aulc_ci95_low, best.aulc_ci95_high)
        ],
        "Final RMSE (MPa)": best.final_rmse_mean.round(2),
    })
    sections = []
    for skew, label in SKEWS.items():
        region = json.loads((output / skew / "protocol.json").read_text())["seed_region"]
        table = summary.loc[summary.skew.eq(skew)].copy()
        table["method"] = table.method.map(NAMES)
        display = table[
            [
                "initial",
                "method",
                "final_rmse_mean",
                "final_rmse_sd",
                "aulc_gain_percent",
                "final_gain_percent",
            ]
        ].round(2)
        display.columns = [
            "Initial labels",
            "Acquisition",
            "Final RMSE",
            "Final SD",
            "Curve gain %",
            "Final gain %",
        ]
        sections.append(f'''<section><h2>{label}</h2><p>{DESCRIPTIONS[skew]} Eligible seed region: {region["eligible"]} of 697 points ({100 * region["fraction"]:.1f}%). Score threshold: {region["threshold"]:.5g}; ties included.</p>
<img src="{skew}/learning_rmse.png" alt="Learning curves for {label}"><p><a href="{skew}/learning_rmse.pdf">Vector PDF</a> · <a href="{skew}/learning_gain.png">Paired gain curves</a> · <a href="{skew}/protocol.json">Protocol</a> · <a href="{skew}/summary.csv">Summary CSV</a></p>
<details><summary>All methods and budgets</summary>{display.to_html(index=False, border=0)}</details></section>''')
    uniform_penalty = compared.loc[compared.method.eq("uniform") & compared["round"].isin([0, 10])]
    penalty_table = (
        uniform_penalty
        .groupby(["skew", "initial", "round"])
        .skew_penalty_mpa.mean()
        .unstack()
        .reset_index()
    )
    penalty_table["skew"] = penalty_table.skew.map(SKEWS)
    penalty_table.columns = [
        "Seed region",
        "Initial labels",
        "Initial RMSE penalty vs IID",
        "Final uniform RMSE penalty vs IID",
    ]
    significant = int(((summary.method.ne("uniform")) & (summary.aulc_ci95_high < 0)).sum())
    audit_record = {
        "campaigns": 540,
        "fitted_evaluators": len(metrics),
        "skews": 4,
        "initial_region_membership": True,
        "same_heldout_ids_as_iid": True,
        "same_data_and_budget_as_iid": True,
        "all_campaigns_complete_before_test_scoring": True,
    }
    (output / "audit.json").write_text(json.dumps(audit_record, indent=2))
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>ICMD · Active learning · Skewed seeds</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1320px;margin:40px auto;padding:0 28px;color:#1b2737;line-height:1.55}}h1{{font-size:36px}}h2{{margin-top:35px}}img{{width:100%}}table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dde3eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{padding:20px;background:#eef3f8;border-radius:8px}}a{{color:#1264a3}}</style></head><body>
<p>QUESTEK / ICMD · OFFLINE ACTIVE LEARNING · SKEWED INITIALIZATION</p><h1>Can active learning recover from a biased seed set?</h1>
<p>Four input-only seed skews × three initial sizes × five paired seeds × nine acquisition methods: <strong>540 campaigns and 5,940 round evaluators</strong>.
The original <a href="{args.iid_url}">IID report is preserved separately</a>.</p>
<p>Only the initial labeled-set distribution changes. Each run still has the full697-point training pool, fixed52 early-stopping points, fixed53 calibration points, and fixed200 test points. Ten rounds acquire50 points each, starting from25,50, or100 labels.</p>
<h2>Comparison with uniform under the same skew</h2><img src="skew_acquisition_gains.png" alt="Acquisition gains by skew and seed size">
<p>Entries are five-seed means of paired percentage improvements in normalized learning-curve area (trapezoidal RMSE over all11 checkpoints, including round0). Positive is better.
{significant} of96 non-uniform comparisons have an unadjusted paired95% t interval entirely below zero for curve-error differences. These are exploratory comparisons without multiple-testing correction.</p>
{best_table.to_html(index=False, border=0)}
<p>The table selects the lowest observed average curve error descriptively; methods and model settings were fixed before these test scores were read. CIs reflect seed variation on a fixed test set, not test-sampling uncertainty.</p>
<h2>How the initial sets differ</h2><img src="seed_feature_shifts.png" alt="Input feature shifts in initial labeled sets">
<p>Regions are defined from eligible training-pool features only, using a fixed35th-percentile score cutoff with ties included. Seed sizes are nested within each seeded permutation of the region. No labels, validation features, or test features define the regions. All outside-region training points remain available for acquisition. The compact region uses the same seven features as the input-distance baseline.</p>
<h2>Initialization penalty and recovery relative to IID</h2>
<p>Positive values mean worse test RMSE than the IID run with the same acquisition method, random seed, and total training-label budget. Below: uniform acquisition, in MPa. Full paired results for every method are downloadable.</p>
{penalty_table.round(2).to_html(index=False, border=0)}
{"".join(sections)}
<h2>Shared methods and interpretation limits</h2><p>The <a href="{args.iid_url}">IID report</a> documents the unchanged CatBoost evaluator, eight-member bootstrap,
ARD Matérn GP uncertainty and sequential GP PV, CoreSet, CQR-width acquisition, and refitted/seed-frozen/random MLP Activation-PV. The same model hyperparameters, feature transform, batches, and seed identifiers are used here.</p>
<div class="note">The fixed representative validation/calibration labels are still available in every run. This isolates skew in the initially acquired training set; it is not a fully cold-start setting with all labeled information biased. Activation-PV uses tabular MLPs without external pretraining. CQR calibration adds a common scalar, so the acquisition rank equals uncalibrated quantile width; adaptive reuse does not provide ordinary conformal coverage guarantees. Conclusions are conditional on this archived, label-available ICMD pool and fixed test split.</div>
<h2>Audit and reproducibility</h2><p>All540 acquisition campaigns finished before scoring test labels. Tests cover label-independent region construction, nested seeds, unchanged held-out identities, full-pool availability, and sequential PV correctness. The reorganized fork reproduced the original IID uniform trajectory's saved predictions exactly.</p>
<p><a href="audit.json">Audit</a> · <a href="suite_protocol.json">Frozen suite protocol</a> · <a href="seed_regions.json">Exact seed regions</a> · <a href="all_summaries.csv">All summaries</a> · <a href="all_metrics.csv">All round metrics</a> · <a href="paired_skew_vs_iid.csv">Paired comparisons with IID</a> · <a href="skew_acquisition_gains.pdf">Overview PDF</a> · <a href="https://github.com/alex-tifrea/alf/tree/codex/icmd-active-learning/benchmark_examples/icmd">Fork code and instructions</a></p>
<p>ALF <a href="https://arxiv.org/pdf/2609.31197">framework paper</a>; <a href="https://arxiv.org/pdf/2605.03964">Activation-PV paper</a>.
Dataset SHA256: <code>{suite["source_sha256"]}</code>.</p></body></html>'''
    (output / "icmd-active-learning-skewed-seeds.html").write_text(html)
    (output / "index.html").write_text(
        '<!doctype html><html><head><title>ICMD · Active learning · Skewed seeds</title><meta http-equiv="refresh" content="0; url=icmd-active-learning-skewed-seeds.html"></head><body><a href="icmd-active-learning-skewed-seeds.html">ICMD · Active learning · Skewed seeds</a></body></html>'
    )
    code = output / "code"
    code.mkdir(exist_ok=True)
    shutil.copytree(
        Path(__file__).parent,
        code / "alf_icmd",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(Path(__file__).parents[2] / "requirements.txt", code / "requirements.txt")
    shutil.copy2(Path(__file__).parents[2] / "README.md", output / "README.md")
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
    print(best_table.to_string(index=False))


if __name__ == "__main__":
    main()
