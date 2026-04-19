"""
generate_outputs.py

Generates all submission artifacts from scored parcel DataFrames:
  - CSV exports (robustness metrics, top-20 shortlists)
  - Matplotlib charts (bar, histogram, scatter)
  - Interactive Folium HTML maps

Usage:
    from src.generate_outputs import save_outputs
    save_outputs(filtered_df, top20_df, version="v3", use_flood_proxy=True)
"""

import matplotlib.pyplot as plt
import folium
import pandas as pd
from pathlib import Path

OUTPUTS_DIR = Path("outputs")
CHARTS_DIR = OUTPUTS_DIR / "charts"


def save_outputs(
    full_df: pd.DataFrame,
    top20_df: pd.DataFrame,
    version: str = "v3",
    use_flood_proxy: bool = True,
) -> None:
    """
    Save all outputs for a given scoring version.

    Args:
        full_df:  All qualified parcels with MC metrics attached.
        top20_df: Top-20 shortlisted parcels.
        version:  Version tag used in filenames (e.g. "v3").
        use_flood_proxy: Whether flood proxy columns are present.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    suffix = f"_v3" if use_flood_proxy else f"_v2"

    # CSV exports
    robustness_cols = [
        "parcel_id", "parcel_area_acres", "centroid_lat", "centroid_lon",
        "dist_to_waterbody_km", "dist_to_flowline_km",
    ]
    if use_flood_proxy:
        robustness_cols += ["flood_proxy_score", "baseline_score_v4"]
    robustness_cols += [
        f"mc_mean_score{suffix}", f"mc_score_std{suffix}",
        f"p_top20{suffix}", f"avg_rank{suffix}",
        f"robust_rank{suffix}", f"confidence_score{suffix}",
    ]
    existing_cols = [c for c in robustness_cols if c in full_df.columns]
    full_df[existing_cols].to_csv(OUTPUTS_DIR / f"robustness_{version}.csv", index=False)
    top20_df.to_csv(OUTPUTS_DIR / f"top20_robust_{version}.csv", index=False)
    print(f"Saved CSVs for version {version}")

    # --- Bar chart: top 20 by MC mean score ---
    mc_col = f"mc_mean_score{suffix}"
    if mc_col in top20_df.columns:
        plot_df = top20_df.sort_values(mc_col, ascending=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(plot_df["parcel_id"].astype(str), plot_df[mc_col])
        ax.set_xlabel("Monte Carlo Mean Score")
        ax.set_ylabel("Parcel ID")
        ax.set_title(f"Top 20 Robust Parcels ({version})")
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / f"top20_robust_{version}_bar.png", dpi=150)
        plt.close(fig)

    # --- Histogram: score stability ---
    std_col = f"mc_score_std{suffix}"
    if std_col in full_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(full_df[std_col], bins=30)
        ax.set_xlabel("Monte Carlo Score Std Dev")
        ax.set_ylabel("Number of Parcels")
        ax.set_title(f"Score Stability Distribution ({version})")
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / f"score_stability_{version}_hist.png", dpi=150)
        plt.close(fig)

    # --- Scatter: confidence vs average rank ---
    rank_col = f"avg_rank{suffix}"
    conf_col = f"confidence_score{suffix}"
    if rank_col in full_df.columns and conf_col in full_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(full_df[rank_col], full_df[conf_col], s=8, alpha=0.5)
        ax.set_xlabel("Average Rank")
        ax.set_ylabel("Confidence Score")
        ax.set_title(f"Confidence vs Average Rank ({version})")
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / f"confidence_vs_rank_{version}.png", dpi=150)
        plt.close(fig)

    print(f"Saved charts for version {version}")

    # --- Interactive Folium map ---
    _save_folium_map(full_df, top20_df, version=version, suffix=suffix,
                     use_flood_proxy=use_flood_proxy)


def _save_folium_map(
    full_df: pd.DataFrame,
    top20_df: pd.DataFrame,
    version: str,
    suffix: str,
    use_flood_proxy: bool,
) -> None:
    map_center = [full_df["centroid_lat"].median(), full_df["centroid_lon"].median()]
    m = folium.Map(location=map_center, zoom_start=10, tiles="OpenStreetMap")

    rank_col = f"robust_rank{suffix}"
    conf_col = f"confidence_score{suffix}"
    p_col = f"p_top20{suffix}"

    for _, row in top20_df.iterrows():
        popup_lines = [
            f"Parcel ID: {row['parcel_id']}",
            f"Robust Rank: {int(row[rank_col]) if rank_col in row.index else 'N/A'}",
            f"Area (acres): {row['parcel_area_acres']:.2f}",
            f"Water Dist (km): {row['dist_to_waterbody_km']:.3f}",
            f"Flowline Dist (km): {row['dist_to_flowline_km']:.3f}",
            f"Top-20 Probability: {row[p_col]:.3f}" if p_col in row.index else "",
            f"Confidence: {row[conf_col]:.3f}" if conf_col in row.index else "",
        ]
        if use_flood_proxy and "flood_proxy_score" in row.index:
            popup_lines.append(f"Flood Proxy: {row['flood_proxy_score']:.3f}")

        folium.CircleMarker(
            location=[row["centroid_lat"], row["centroid_lon"]],
            radius=7,
            popup="<br>".join(filter(None, popup_lines)),
            fill=True,
            color="crimson",
            fill_opacity=0.7,
        ).add_to(m)

    map_path = OUTPUTS_DIR / f"map_robust_top20_{version}.html"
    m.save(str(map_path))
    print(f"Saved interactive map -> {map_path}")
