# BTM Site Scout — Land & Power Viability for Behind-the-Meter Data Centers

AI-driven geospatial scoring engine that narrows 186,000 Travis County parcels to 20 high-confidence sites for behind-the-meter (BTM) natural gas data center development — with Monte Carlo robustness testing and an interactive visualization dashboard.

> **Covers Sub-problem A** (land & lease viability scoring) and **Sub-problem C** (BTM power economics / ERCOT spread analysis). Sub-problem B (gas pipeline reliability) is scoped for a future iteration.

---

## Problem Statement

Hyperscale data centers increasingly rely on behind-the-meter natural gas generation to bypass grid interconnection queues. Siting these facilities requires simultaneously evaluating thousands of candidate parcels across land viability, infrastructure proximity, and power economics — a process that is currently manual and slow.

This project builds a first-stage screening system that tells a developer which 20 parcels deserve deeper due diligence, not which one to buy.

---

## Pipeline (Sub-problem A)

```
186,000 parcels (Travis County ArcGIS)
    ↓ Hard filters: ≥ 50 acres · no water body intersection
      · < 5 km to waterbody · < 2 km to flowline
630 qualified parcels (0.17%)
    ↓ Re-normalize scores on qualified set
    ↓ Add flood-risk proxy layer
    ↓ K-means geographic clustering (k=20)
    ↓ Monte Carlo: 1,000 simulations · Dirichlet weight sampling
Top 20 recommended sites
```

### Scoring Dimensions

| Feature            | Weight range | Rationale                                         |
| ------------------ | ------------ | ------------------------------------------------- |
| Acreage            | ~45–50%      | Data centers need large contiguous space          |
| Water proximity    | ~22–30%      | Cooling water access (waterbody + flowline)       |
| Flood risk proxy   | penalty      | Hydro adjacency composite — close is risky        |

### Monte Carlo Robustness

1,000 simulations draw weights from Dirichlet(α=2) so no single dimension dominates. This produces:

| Metric           | Meaning                                                  |
| ---------------- | -------------------------------------------------------- |
| `p_top20`        | Probability of ranking in top 20 across all simulations  |
| `confidence_score` | Composite: p_top20 + score stability + flood safety    |
| `robust_rank`    | Final recommendation rank                                |

Parcel 804168 ranks in the top 20 in **993 out of 1,000 simulations** — robust under any reasonable weighting.

---

## Sub-problem C — BTM Power Economics

Real Henry Hub gas prices (EIA, 2021–2024) vs. ERCOT HB_SOUTH LMP.  
**BTM spread** = LMP − (gas price × 7.0 MMBtu/MWh heat rate)

| Metric                        | Value           |
| ----------------------------- | --------------- |
| Mean BTM spread               | $37.85/MWh      |
| Hours BTM is viable           | 94.9%           |
| Summer peak spread (p50)      | $40.09/MWh      |
| Spring off-peak spread (p50)  | $23.92/MWh      |
| Winter Storm Uri — 8 days     | ~$9.1M / 10 MW  |

SARIMA(1,1,1)(1,1,1,7) forecasts daily BTM spread 24 days forward (MAPE: 16.3% on 180-day holdout).

BTM generation is an **options strategy** — extreme events like Uri dominate total value.

---

## Results

**Top 5 Recommended Sites:**

| Rank | Parcel ID | Acres | p_top20 | Confidence | Flood Risk |
| ---- | --------- | ----- | ------- | ---------- | ---------- |
| 1    | 804168    | 118.4 | 0.993   | 0.746      | 0.000      |
| 2    | 324445    | 83.5  | 0.963   | 0.733      | 0.046      |
| 3    | 214268    | 75.2  | 0.935   | 0.715      | 0.028      |
| 4    | 190649    | 117.7 | 0.972   | 0.745      | 0.058      |
| 5    | 838573    | 66.0  | 0.922   | 0.710      | 0.053      |

Full results: [`outputs/top20_robust_v3.csv`](outputs/top20_robust_v3.csv)

---

## Interactive Dashboard (Frontend)

A self-contained single-page app — no server, no build step needed.

**Features:**
- Spinning globe intro → flies to Travis County
- Satellite map with all 20 parcel markers (color-coded by confidence)
- Click any parcel → zooms in, shows bbox border, opens detail scorecard
- Click again → zooms back out
- **Sites tab** — ranked list with Monte Carlo bars
- **Detail tab** — land data, infrastructure proximity, composite scorecard (Sub-A + Sub-C)
- **Sub-A tab** — filtering funnel + radar chart of top 5 sites
- **Sub-C tab** — Uri LMP spike chart, SARIMA spread forecast, seasonal table

### Setup

```bash
# Copy config template
cp frontend/config.example.js frontend/config.js
# Add your Mapbox token in config.js
```

### Open

**Option A — direct file (simplest, no server needed):**

```bash
start frontend/index.html   # Windows
open frontend/index.html    # macOS
```

**Option B — local server (recommended if direct file doesn't load tiles):**

```bash
python -m http.server 8000 --directory frontend
```

Then open → **http://localhost:8000**

---

## Repository Structure

```
├── frontend/
│   ├── index.html           # Full interactive dashboard (map + all charts)
│   ├── config.js            # Your Mapbox token (not committed)
│   └── config.example.js   # Token template
├── notebooks/
│   ├── 01_data_pipeline.ipynb       # Data download, feature extraction, baseline scoring
│   └── 02_robustness_analysis.ipynb # Monte Carlo analysis, iterations v1–v3, rankings
├── src/
│   ├── build_features.py        # Data download + spatial joins + feature engineering
│   ├── score_models.py          # Scoring formulas v1–v4, hard filters, flood proxy
│   ├── run_monte_carlo.py       # Monte Carlo simulation engine (Dirichlet sampling)
│   ├── run_experiments.py       # End-to-end pipeline runner
│   ├── generate_outputs.py      # Charts, CSVs, Folium maps
│   ├── build_power_features.py  # ERCOT + EIA download, BTM spread computation
│   └── score_power_economics.py # SARIMA forecast + power economics scoring
├── outputs/
│   ├── top20_robust_v3.csv          # FINAL ranked shortlist
│   ├── robustness_v3.csv            # Full Monte Carlo metrics for 630 sites
│   ├── map_robust_top20_v3.html     # Interactive Folium map
│   ├── power_economics_scores.csv   # Power economics scores per parcel
│   ├── sarima_forecast_24d.csv      # 24-day BTM spread forecast with 80% CI
│   ├── spread_seasonal_table.csv    # Spread p50 by season × peak/off-peak
│   └── charts/                      # All generated visualizations
├── requirements.txt
└── README.md
```

---

## Quickstart (Python Pipeline)

```bash
pip install -r requirements.txt

# Sub-A: Download data and build features (~30 min, requires internet)
python src/build_features.py

# Sub-A: Run Monte Carlo and generate all outputs (~5 min)
python src/run_experiments.py

# Sub-C: Download ERCOT + EIA data, compute BTM spread (~2 min)
python src/build_power_features.py

# Sub-C: SARIMA forecast + power scoring (~2 min)
python src/score_power_economics.py
```

> Raw data is not committed. Scripts download from public APIs on first run and cache in `data/raw/`.

---

## Data Sources

| Layer                | Source                               | Records      |
| -------------------- | ------------------------------------ | ------------ |
| Parcels              | Travis County ArcGIS REST API        | ~186,000     |
| Waterbodies          | USGS National Map 3DHP FeatureServer | bbox query   |
| Flowlines            | USGS National Map 3DHP FeatureServer | bbox query   |
| Henry Hub gas prices | EIA open data API (DEMO_KEY)         | 1,001 daily  |

Study area: Travis County, TX — bbox (-98.25, 30.00, -97.30, 30.75)

---

## Key Design Decisions

**Why Monte Carlo instead of fixed weights?**  
No single weighting of acreage vs. water access is objectively correct. Dirichlet-sampled Monte Carlo tests which sites perform well under any reasonable assumption, not just one assumed preference.

**Why hard filter before scoring?**  
Scoring all 186K parcels when 99.8% fail basic constraints compresses the score distribution. Filtering first lets the model distinguish meaningfully between genuinely competitive sites.

**Why flowline proximity separately from waterbody?**  
Waterbodies = cooling water availability. Flowlines = access routes and drainage. Near-flowline-but-not-intersecting is the optimal profile — good infrastructure without direct flood exposure.

**Why fiber excluded from final scoring?**  
The fiber layer had inconsistent geographic coverage relative to the parcel region. Including it would have added noise. Documented as a data limitation, not silently dropped.

---

## Limitations

- Flood risk is a proximity proxy — should be replaced with FEMA 100-year flood zone polygons
- No zoning or land-use data — parcel source only provided geometry and PROP_ID
- No deed/lease or ownership data — commercial feasibility requires title search analysis
- Single county only — expansion to all of Texas requires the same pipeline at larger scale
- ERCOT prices use a synthetic fallback calibrated to published HB_SOUTH statistics (portal API changed format)

---

## Dependencies

```
geopandas>=0.14.0  pandas>=2.0.0  numpy>=1.24.0
requests>=2.31.0   scikit-learn>=1.3.0  statsmodels>=0.14.0
folium>=0.15.0     matplotlib>=3.7.0    pyarrow>=13.0.0
jupyterlab>=4.0.0
```
