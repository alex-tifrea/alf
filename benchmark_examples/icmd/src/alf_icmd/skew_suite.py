"""Run four prespecified feature-skewed initializations with an unchanged AL protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from alf_icmd.constants import METHODS
from alf_icmd.data import SKEWS
from alf_icmd.report import audit, summarize
from alf_icmd.runner import evaluate, plot, prepare, run_one


def main():
    """Finish every acquisition campaign before evaluating any skewed test predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = argparse.Namespace(
        initial_sizes=[25, 50, 100],
        seeds=list(range(20261027, 20261032)),
        methods=METHODS,
        rounds=10,
        batch=50,
        seed_skew="iid",
    )
    suite = {
        "skews": list(SKEWS),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "initial_sizes": config.initial_sizes,
        "seeds": config.seeds,
        "methods": METHODS,
        "rounds": config.rounds,
        "batch": config.batch,
        "skew_quantile": 0.35,
        "test_policy": "score only after all 540 acquisition campaigns finish",
    }
    lock = args.output / "suite_protocol.json"
    if lock.exists():
        assert json.loads(lock.read_text()) == suite
    lock.write_text(json.dumps(suite, indent=2))
    jobs = []
    for skew in SKEWS:
        config.seed_skew = skew
        directory = args.output / skew
        prepare(args.source, directory, config)
        jobs.extend(
            (
                str(args.source),
                str(directory),
                method,
                seed,
                initial,
                config.rounds,
                config.batch,
                skew,
            )
            for initial in config.initial_sizes
            for seed in config.seeds
            for method in METHODS
        )
    if not args.report_only:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for number, result in enumerate(executor.map(run_one, jobs), 1):
                print(f"{number}/{len(jobs)} {result}", flush=True)
    assert len(list(args.output.glob("*/runs/*/complete.json"))) == len(jobs)
    for skew in SKEWS:
        directory = args.output / skew
        _, summary, paired = evaluate(args.source, directory)
        plot(directory, summary, paired)
        audit(directory)
        summarize(directory)
        print(f"Evaluated {skew}", flush=True)


if __name__ == "__main__":
    main()
