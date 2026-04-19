"""
run_monte_carlo.py

Monte Carlo robustness analysis for parcel scoring.

Runs N simulations where scoring weights are randomly sampled from a Dirichlet
distribution. For each parcel this produces:
  - mc_mean_score   — mean score across simulations
  - mc_score_std    — score volatility (lower = more stable)
  - p_top20         — fraction of simulations where parcel ranked in top 20
  - avg_rank        — average rank across simulations
  - robust_rank     — final rank (ascending avg_rank)
  - confidence_score — composite robustness index

Usage:
    from src.run_monte_carlo import run_monte_carlo
    results = run_monte_carlo(filtered_df)
"""

import numpy as np
import pandas as pd


def run_monte_carlo(
    df: pd.DataFrame,
    n_simulations: int = 1000,
    top_k: int = 20,
    seed: int = 42,
    use_flood_proxy: bool = False,
) -> pd.DataFrame:
    """
    Monte Carlo weight sensitivity analysis.

    Positive weights (acreage, water, flowline) are drawn from a symmetric
    Dirichlet(alpha=2) distribution so they sum to 1 and no dimension
    dominates. Penalty weights are drawn independently from uniform ranges.

    Args:
        df: DataFrame with normalized score columns (acreage_score, water_score,
            flowline_score, water_intersection_penalty, and optionally
            flood_proxy_score / flood_proxy_penalty).
        n_simulations: Number of Monte Carlo draws.
        top_k: Threshold for computing p_topK metric.
        seed: Random seed for reproducibility.
        use_flood_proxy: If True, include flood_proxy_score as an additional
            penalty term (requires flood_proxy_score column in df).

    Returns:
        df with added columns: mc_mean_score, mc_score_std, p_top{k},
        avg_rank, robust_rank, confidence_score.
    """
    suffix = "_v3" if use_flood_proxy else "_v2"
    rng = np.random.default_rng(seed)

    acreage = df["acreage_score"].values
    water = df["water_score"].values
    flowline = df["flowline_score"].values
    penalty = df["water_intersection_penalty"].values
    flood_proxy = df["flood_proxy_score"].values if use_flood_proxy else None

    n = len(df)
    score_matrix = np.zeros((n, n_simulations), dtype=np.float32)
    topk_counts = np.zeros(n, dtype=np.int32)
    rank_sum = np.zeros(n, dtype=np.float64)

    for i in range(n_simulations):
        # Positive weights: Dirichlet ensures they sum to 1
        pos_w = rng.dirichlet([2.0, 2.0, 2.0])

        sim_score = (
            pos_w[0] * acreage
            + pos_w[1] * water
            + pos_w[2] * flowline
        )

        if use_flood_proxy:
            water_penalty_w = rng.uniform(0.08, 0.20)
            flood_proxy_w = rng.uniform(0.05, 0.15)
            sim_score -= water_penalty_w * penalty
            sim_score -= flood_proxy_w * flood_proxy
        else:
            penalty_w = rng.uniform(0.15, 0.35)
            sim_score -= penalty_w * penalty

        score_matrix[:, i] = sim_score

        order = np.argsort(-sim_score)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n + 1)
        rank_sum += ranks
        topk_counts += (ranks <= top_k).astype(np.int32)

    result = df.copy()
    result[f"mc_mean_score{suffix}"] = score_matrix.mean(axis=1)
    result[f"mc_score_std{suffix}"] = score_matrix.std(axis=1)
    result[f"p_top{top_k}{suffix}"] = topk_counts / n_simulations
    result[f"avg_rank{suffix}"] = rank_sum / n_simulations
    result[f"robust_rank{suffix}"] = (
        result[f"avg_rank{suffix}"].rank(method="dense").astype(int)
    )

    # Confidence score: combines top-k probability, score stability, and safety signals
    std_col = result[f"mc_score_std{suffix}"]
    std_scaled = (std_col - std_col.min()) / (std_col.max() - std_col.min() + 1e-9)

    if use_flood_proxy:
        result[f"confidence_score{suffix}"] = (
            0.55 * result[f"p_top{top_k}{suffix}"]
            + 0.25 * (1 - std_scaled)
            + 0.10 * (1 - result["water_intersection_penalty"])
            + 0.10 * (1 - result["flood_proxy_penalty"])
        )
    else:
        result[f"confidence_score{suffix}"] = (
            0.60 * result[f"p_top{top_k}{suffix}"]
            + 0.30 * (1 - std_scaled)
            + 0.10 * (1 - result["water_intersection_penalty"])
        )

    return result


def get_top_k(df: pd.DataFrame, k: int = 20, use_flood_proxy: bool = False) -> pd.DataFrame:
    """Return the top-k parcels sorted by robust_rank, then p_top20, then mc_mean_score."""
    suffix = "_v3" if use_flood_proxy else "_v2"
    return df.sort_values(
        [f"robust_rank{suffix}", f"p_top20{suffix}", f"mc_mean_score{suffix}"],
        ascending=[True, False, False],
    ).head(k).copy()
