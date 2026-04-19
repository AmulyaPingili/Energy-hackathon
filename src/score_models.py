"""
score_models.py

Scoring model definitions for BTM data center parcel suitability.

Scoring evolution:
  v1 — Baseline weighted sum (acreage, water, flowline, intersection penalty)
  v2 — Upweighted penalty for water intersection
  v3 — Post-filter rescoring on qualified parcels only
  v4 — Adds flood-risk proxy layer as additional penalty

All scores are in [0, 1]. Higher = better candidate site.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hard filter thresholds
# ---------------------------------------------------------------------------
MIN_ACRES = 50          # Minimum contiguous acreage for a viable data center pad
MAX_WATERBODY_KM = 5    # Must be within 5 km of cooling water source
MAX_FLOWLINE_KM = 2     # Must be within 2 km of a flowline


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def minmax_scale(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]. Handles constant series."""
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def add_normalized_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and attach normalized score columns to df (in-place copy).
    Assumes df contains: parcel_area_acres, dist_to_waterbody_km,
    dist_to_flowline_km, intersects_waterbody.
    """
    df = df.copy()
    df["acreage_score"] = minmax_scale(df["parcel_area_acres"])
    df["water_score"] = 1 - minmax_scale(df["dist_to_waterbody_km"])
    df["flowline_score"] = 1 - minmax_scale(df["dist_to_flowline_km"])
    df["water_intersection_penalty"] = df["intersects_waterbody"].astype(float)
    return df


# ---------------------------------------------------------------------------
# Hard filter
# ---------------------------------------------------------------------------

def apply_hard_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove parcels that fail minimum viability thresholds.

    Thresholds:
      - Area >= MIN_ACRES (50 acres)
      - No direct water body intersection
      - Within MAX_WATERBODY_KM of a water source
      - Within MAX_FLOWLINE_KM of a flowline

    Returns the filtered DataFrame (rows that pass all checks).
    """
    mask = (
        (df["parcel_area_acres"] >= MIN_ACRES)
        & (df["intersects_waterbody"] == 0)
        & (df["dist_to_waterbody_km"] <= MAX_WATERBODY_KM)
        & (df["dist_to_flowline_km"] <= MAX_FLOWLINE_KM)
    )
    filtered = df[mask].dropna().copy()
    print(f"Hard filter: {len(df):,} -> {len(filtered):,} parcels ({len(filtered)/len(df)*100:.2f}%)")
    return filtered


# ---------------------------------------------------------------------------
# Scoring models
# ---------------------------------------------------------------------------

def score_v1(df: pd.DataFrame) -> pd.Series:
    """
    Baseline score — broad initial ranking of all parcels.
    Weights: acreage 40%, water proximity 30%, flowline 20%, intersection penalty 10%.
    """
    return (
        0.40 * df["acreage_score"]
        + 0.30 * df["water_score"]
        + 0.20 * df["flowline_score"]
        - 0.10 * df["water_intersection_penalty"]
    )


def score_v2(df: pd.DataFrame) -> pd.Series:
    """
    Stronger water intersection penalty. Used for initial shortlisting.
    Weights: acreage 45%, water 30%, flowline 15%, intersection penalty 25%.
    """
    return (
        0.45 * df["acreage_score"]
        + 0.30 * df["water_score"]
        + 0.15 * df["flowline_score"]
        - 0.25 * df["water_intersection_penalty"]
    )


def score_v3(df: pd.DataFrame) -> pd.Series:
    """
    Post-filter baseline. Rescored on the 630-parcel qualified set only,
    so normalization reflects the competitive range of viable sites.
    Weights: acreage 50%, water 30%, flowline 20%, intersection penalty 25%.
    """
    return (
        0.50 * df["acreage_score"]
        + 0.30 * df["water_score"]
        + 0.20 * df["flowline_score"]
        - 0.25 * df["water_intersection_penalty"]
    )


def add_flood_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a flood-risk proxy score and binary high-risk flag.

    Proxy formula (normalized to [0,1]):
      flood_proxy = 0.45 * (1 - water_score)   # closer to waterbody => higher risk
                  + 0.35 * (1 - flowline_score) # closer to flowline => higher risk
                  + 0.20 * intersects_waterbody # direct intersection => maximum risk

    A parcel is flagged flood_proxy_penalty=1 if flood_proxy >= 0.70.
    """
    df = df.copy()
    raw_proxy = (
        0.45 * (1 - df["water_score"])
        + 0.35 * (1 - df["flowline_score"])
        + 0.20 * df["water_intersection_penalty"]
    )
    df["flood_proxy_score"] = minmax_scale(raw_proxy)
    df["flood_proxy_penalty"] = (df["flood_proxy_score"] >= 0.70).astype(float)
    return df


def score_v4(df: pd.DataFrame) -> pd.Series:
    """
    Full score with flood proxy. Final deterministic ranking before Monte Carlo.
    Weights: acreage 45%, water 22%, flowline 13%, intersection penalty 10%,
             flood proxy penalty 10%.
    """
    return (
        0.45 * df["acreage_score"]
        + 0.22 * df["water_score"]
        + 0.13 * df["flowline_score"]
        - 0.10 * df["water_intersection_penalty"]
        - 0.10 * df["flood_proxy_score"]
    )
