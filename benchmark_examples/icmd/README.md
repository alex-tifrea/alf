# ICMD offline active learning

This example compares nine acquisitions for yield-strength regression through ALF's
`DesignTask`, `Optimizer`, and offline `Oracle`. It uses a common CatBoost evaluator
so acquisition quality is compared at the same training-label budget.

## Install and run

From this directory, `uv sync` installs the benchmark and the local `../../core`
checkout. It does not require the unrelated protein or molecular dependencies.

```bash
uv sync
uv run pytest
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run alf-icmd \
  --source /absolute/path/to/dataset.parquet \
  --output "$HOME/.reports/icmd-iid" --workers 3
uv run alf-icmd-report \
  --source /absolute/path/to/dataset.parquet \
  --output "$HOME/.reports/icmd-iid"
```

The input is a prepared Parquet table with `query_id`, `split` (`train`, `val`,
`test`), `yield_strength_mpa`, and the six physical inputs in `constants.py`.
Input data and generated reports are deliberately not committed. Supply the
ICMD table through your authorized storage access. The original dataset hash is
`7dbb4c73a09c4c47d23a9bab34c925fd5b0aaf37f84a448185017df2130683f5`.

## Frozen comparison

- 697 eligible training/pool records; original 105 validation and 200 test records.
- Fixed validation partition: 52 for early stopping, 53 for conformal calibration.
- Initial sizes 25, 50, 100, nested within each of five seeds (20261027–20261031).
- Ten rounds of 50 acquisitions; all methods share each seed's starting set.
- Uniform, input CoreSet, eight-member bootstrap uncertainty, GP marginal variance,
  sequential GP PV, CQR interval width, and refitted/frozen/random MLP Activation-PV.
- Test split labels are masked during the AL loop. Test-input predictions are
  scored after all prespecified campaigns finish. The raw offline oracle retains
  archived labels; acquisition implementations do not access them.

The runner writes a protocol lock, split identities, per-round acquired IDs,
predictions, raw metrics, paired differences, and PNG/PDF curves. The report
builder audits pairing and split isolation and writes `icmd-active-learning.html`
with an `index.html` compatibility redirect. Model and acquisition parameters are
recorded in `protocol.json`. Output directories are resumable only for an unchanged
protocol. `--report-only` rebuilds figures without retraining.

## Interpretation

Activation-PV is a tabular adaptation using 7–64–64–1 MLP hidden features, not a
replication with a pretrained atomic model. CQR's additive calibration correction
is shared across candidates, so its ranking equals raw quantile-width ranking.
Repeated adaptive reuse of calibration data does not provide the usual conformal
coverage guarantee. Fixed validation/calibration labels are shared overhead beyond
the plotted acquired-training-label budget. Error bars describe seed variation
conditional on the fixed data split.

## Layout

- `data.py`: schema-preserving split and oracle adapter.
- `models.py`: common evaluator and MLP feature extractor.
- `acquisition.py`: input-only scoring and sequential batch selection.
- `runner.py`: campaigns, immutable protocol, deferred scoring, and figures.
- `report.py`: audits, paired summaries, export, and readable report.
- `tests/`: split isolation, seed nesting, and explicit PV-conditioning equivalence.

The baseline uses upstream ALF revision
`2002936c623d11bf58f33b8cfb198ecc59b4c4aa`. See the
[ALF paper](https://arxiv.org/pdf/2609.31197) and
[activation-kernel PV paper](https://arxiv.org/pdf/2605.03964).
