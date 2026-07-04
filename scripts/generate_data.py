#!/usr/bin/env python
"""Generate synthetic demand datasets locally for prototyping + offline runs.

Reuses `forecasting.data.generator` (the SAME code the notebook and tests use)
so local data is identical in schema to what the deployed BigQuery mart yields.

Examples
--------
    # Default: 365-day base + 365-day drifted window -> data/raw/*.csv
    python scripts/generate_data.py

    # Base only (no drift), 500 days, custom seed
    python scripts/generate_data.py --mode base --n-days 500 --seed 11

    # Base + drift with a stronger level shock, write parquet too
    python scripts/generate_data.py --mode drift --level-scale 1.6 --parquet

    # Also emit a CSV ready to `bq load` into demand_raw.sales
    python scripts/generate_data.py --bq-load-hint
"""

from __future__ import annotations

import argparse
from pathlib import Path

from forecasting.data.generator import (
    DriftConfig,
    SeriesConfig,
    generate_base_and_drift,
    generate_series,
)

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["base", "drift"],
        default="drift",
        help="'base' = single regime; 'drift' = base + drifted window (default).",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--n-days", type=int, default=365)
    p.add_argument("--base-level", type=float, default=80.0)
    p.add_argument("--noise-std", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)

    # Drift knobs (only used when --mode drift)
    p.add_argument(
        "--level-scale",
        type=float,
        default=1.35,
        help="Multiply demand level in the drifted window.",
    )
    p.add_argument(
        "--extra-trend",
        type=float,
        default=0.15,
        help="Added daily trend in the drifted window.",
    )
    p.add_argument(
        "--noise-scale",
        type=float,
        default=2.0,
        help="Scale noise std in the drifted window.",
    )
    p.add_argument("--drift-seed", type=int, default=7)

    p.add_argument(
        "--parquet", action="store_true", help="Also write parquet alongside CSV."
    )
    p.add_argument(
        "--bq-load-hint",
        action="store_true",
        help="Print a ready-to-run `bq load` command.",
    )
    return p.parse_args(argv)


def _write(df, out_dir: Path, name: str, parquet: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[write] {csv_path}  ({len(df)} rows)")
    if parquet:
        pq_path = out_dir / f"{name}.parquet"
        df.to_parquet(pq_path, index=False)
        print(f"[write] {pq_path}")
    return csv_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    base_cfg = SeriesConfig(
        start_date=args.start_date,
        n_days=args.n_days,
        base_level=args.base_level,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    if args.mode == "base":
        df = generate_series(base_cfg)
        primary = _write(df, args.out_dir, "sales", args.parquet)
    else:
        drift_cfg = DriftConfig(
            level_scale=args.level_scale,
            extra_trend_per_day=args.extra_trend,
            noise_scale=args.noise_scale,
            seed=args.drift_seed,
        )
        base_df, drift_df, full_df = generate_base_and_drift(base_cfg, drift_cfg)
        _write(base_df, args.out_dir, "sales_base", args.parquet)
        _write(drift_df, args.out_dir, "sales_drift", args.parquet)
        # `sales` = the full union the pipeline ingests (base + drift).
        primary = _write(full_df, args.out_dir, "sales", args.parquet)

    if args.bq_load_hint:
        print(
            "\n# Load into BigQuery Sandbox (raw source table):\n"
            f"bq load --autodetect --replace --source_format=CSV \\\n"
            f"    demand_raw.sales {primary}\n"
        )
    print("[done] dataset generation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
