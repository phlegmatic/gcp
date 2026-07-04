"""Synthetic demand dataset generation (base series + injectable drift).

Lives in the installable package so BOTH the CLI script (`scripts/generate_data.py`)
and the local notebook reuse the exact same logic -- no divergence between what
you prototype locally and what the deployed pipeline consumes.

Output schema matches the dbt raw source `demand_raw.sales`:
    sale_date : DATE
    units_sold: FLOAT

The same frame renamed to (`ds`, `demand`) is what the pipeline mart produces,
so downstream `forecasting.models.features.build_features` consumes it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RAW_DATE_COL = "sale_date"
RAW_TARGET_COL = "units_sold"


@dataclass
class SeriesConfig:
    """Parameters describing a single synthetic demand regime."""

    start_date: str = "2023-01-01"
    n_days: int = 365
    base_level: float = 80.0
    trend_per_day: float = 0.05
    weekly_amplitude: float = 12.0
    yearly_amplitude: float = 20.0
    noise_std: float = 4.0
    #: Multiplicative demand spikes on these ISO weekdays (0=Mon..6=Sun).
    weekend_uplift: float = 1.15
    #: Random promo spikes: probability per day and multiplier.
    promo_prob: float = 0.03
    promo_multiplier: float = 1.6
    seed: int = 42


@dataclass
class DriftConfig:
    """Overrides applied to a SeriesConfig to simulate distribution drift.

    Each field, when not None, replaces / scales the base behavior so you can
    reproduce realistic covariate + concept drift that Evidently will flag.
    """

    #: Multiply the whole level (demand shock / market change).
    level_scale: float = 1.35
    #: Add to the daily trend (accelerating/decelerating growth).
    extra_trend_per_day: float = 0.15
    #: Scale seasonality amplitude (changing seasonal behavior).
    weekly_amplitude_scale: float = 0.5
    #: Scale noise (regime becomes more/less volatile).
    noise_scale: float = 2.0
    #: Extra promo frequency in the drifted window.
    promo_prob: float = 0.10
    seed: int = 7
    tags: list[str] = field(default_factory=lambda: ["drift"])


def _seasonal_component(dates: pd.DatetimeIndex, cfg: SeriesConfig) -> np.ndarray:
    day_of_week = dates.dayofweek.to_numpy()
    day_of_year = dates.dayofyear.to_numpy()
    weekly = cfg.weekly_amplitude * np.sin(2 * np.pi * day_of_week / 7.0)
    yearly = cfg.yearly_amplitude * np.sin(2 * np.pi * day_of_year / 365.0)
    return weekly + yearly


def generate_series(cfg: SeriesConfig) -> pd.DataFrame:
    """Generate a base demand series as a raw (sale_date, units_sold) frame."""
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start_date, periods=cfg.n_days, freq="D")

    trend = cfg.base_level + cfg.trend_per_day * np.arange(cfg.n_days)
    seasonal = _seasonal_component(dates, cfg)
    noise = rng.normal(0.0, cfg.noise_std, cfg.n_days)

    demand = trend + seasonal + noise

    # Weekend uplift (multiplicative).
    is_weekend = dates.dayofweek.to_numpy() >= 5
    demand = np.where(is_weekend, demand * cfg.weekend_uplift, demand)

    # Random promotional spikes.
    promo = rng.random(cfg.n_days) < cfg.promo_prob
    demand = np.where(promo, demand * cfg.promo_multiplier, demand)

    demand = np.clip(demand, a_min=0.0, a_max=None)

    return pd.DataFrame({RAW_DATE_COL: dates.date, RAW_TARGET_COL: demand.round(2)})


def _apply_drift(
    base: SeriesConfig, drift: DriftConfig, start_date: str
) -> SeriesConfig:
    """Return a new SeriesConfig representing the drifted regime."""
    return SeriesConfig(
        start_date=start_date,
        n_days=base.n_days,
        base_level=base.base_level * drift.level_scale,
        trend_per_day=base.trend_per_day + drift.extra_trend_per_day,
        weekly_amplitude=base.weekly_amplitude * drift.weekly_amplitude_scale,
        yearly_amplitude=base.yearly_amplitude,
        noise_std=base.noise_std * drift.noise_scale,
        weekend_uplift=base.weekend_uplift,
        promo_prob=drift.promo_prob,
        promo_multiplier=base.promo_multiplier,
        seed=drift.seed,
    )


def generate_base_and_drift(
    base_cfg: SeriesConfig | None = None,
    drift_cfg: DriftConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a reference (base) window, a drifted window, and their union.

    Returns
    -------
    (base_df, drift_df, full_df)
        `base_df`  : the reference regime (earlier dates).
        `drift_df` : the drifted regime, starting the day after base ends.
        `full_df`  : base + drift concatenated -- feed this to the pipeline so
                     the drift step compares the two halves.
    """
    base_cfg = base_cfg or SeriesConfig()
    drift_cfg = drift_cfg or DriftConfig()

    base_df = generate_series(base_cfg)

    last_base_date = pd.to_datetime(base_df[RAW_DATE_COL].iloc[-1])
    drift_start = (last_base_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    drifted_series_cfg = _apply_drift(base_cfg, drift_cfg, drift_start)
    drift_df = generate_series(drifted_series_cfg)

    full_df = (
        pd.concat([base_df, drift_df], ignore_index=True)
        .sort_values(RAW_DATE_COL)
        .reset_index(drop=True)
    )
    return base_df, drift_df, full_df


def to_pipeline_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Rename a raw (sale_date, units_sold) frame to the pipeline (ds, demand)."""
    return raw_df.rename(columns={RAW_DATE_COL: "ds", RAW_TARGET_COL: "demand"})
