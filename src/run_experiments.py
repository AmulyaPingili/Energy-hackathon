"""
run_experiments.py

End-to-end experiment runner for BTM data center site selection.

Executes the full pipeline from processed features through to final ranked outputs:
  1. Load features_clean.parquet (built by build_features.py)
  2. Apply hard filters (acreage, water distance, flood risk)
  3. Recompute normalized scores on the filtered subset
  4. Add flood-risk proxy layer
  5. Run K-means geographic clustering
  6. Run Monte Carlo robustness analysis (1,000 simulations)
  7. Save all outputs (CSVs, charts, interactive maps)

Usage:
    python src/run_experiments.py

Prerequisites:
    Run src/build_features.py first to generate data/processed/features_clean.parquet.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

from score_models import (
    apply_hard_filters,
    add_normalized_scores,
    add_flood_proxy,
    score_v3,
    score_v4,
)
from run_monte_carlo import run_monte_carlo, get_top_k
from generate_outputs import save_outputs

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")
N_CLUSTERS = 20
N_SIMULATIONS = 1000
RANDOM_SEED = 42


def add_clusters(df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    """K-means cluster parcels by geographic location for diversity analysis."""
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    df = df.copy()
    df["location_cluster"] = km.fit_predict(df[["centroid_lat", "centroid_lon"]])

    cluster_stats = (
        df.groupby("location_cluster")
        .agg(
            cluster_size=("parcel_id", "count"),
            cluster_avg_area=("parcel_area_acres", "mean"),
            cluster_avg_water_dist=("dist_to_waterbody_km", "mean"),
            cluster_avg_flow_dist=("dist_to_flowline_km", "mean"),
        )
        .reset_index()
    )
    return df.merge(cluster_stats, on="location_cluster", how="left")


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load features
    features_path = PROCESSED_DIR / "features_clean.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} not found. Run src/build_features.py first."
        )
    feature_df = pd.read_parquet(features_path)
    print(f"Loaded {len(feature_df):,} parcels from {features_path}")

    # 2. Hard filter
    filtered = apply_hard_filters(feature_df)

    # 3. Re-normalize scores on the qualified subset (so ranges reflect viable sites only)
    filtered = add_normalized_scores(filtered)

    # 4. Baseline v3 score
    filtered["baseline_score_v3"] = score_v3(filtered)

    # 5. Flood proxy + baseline v4
    filtered = add_flood_proxy(filtered)
    filtered["baseline_score_v4"] = score_v4(filtered)

    # 6. Geographic clustering
    filtered = add_clusters(filtered)

    # 7. Monte Carlo (with flood proxy)
    print(f"Running {N_SIMULATIONS} Monte Carlo simulations...")
    filtered = run_monte_carlo(
        filtered,
        n_simulations=N_SIMULATIONS,
        top_k=20,
        seed=RANDOM_SEED,
        use_flood_proxy=True,
    )

    top20 = get_top_k(filtered, k=20, use_flood_proxy=True)

    # 8. Save outputs
    filtered.to_parquet(PROCESSED_DIR / "features_filtered_v3.parquet", index=False)
    save_outputs(filtered, top20, version="v3", use_flood_proxy=True)

    print("\n=== Top 5 Recommended Sites ===")
    display_cols = [
        "parcel_id", "parcel_area_acres",
        "dist_to_waterbody_km", "dist_to_flowline_km",
        "flood_proxy_score", "p_top20_v3", "confidence_score_v3", "robust_rank_v3",
    ]
    available = [c for c in display_cols if c in top20.columns]
    print(top20[available].head(5).to_string(index=False))
    print(f"\nFull results saved to outputs/top20_robust_v3.csv")


if __name__ == "__main__":
    main()
