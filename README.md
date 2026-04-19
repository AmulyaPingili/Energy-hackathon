# BTM Data Center Site Selection — Sub-problem A: Land & Lease Viability

AI-driven geospatial scoring engine for identifying optimal land parcels for behind-the-meter (BTM) data center development in Austin, TX (Travis County).

> **Hackathon scope:** This submission addresses Sub-problem A only (land & lease viability scoring). Sub-problems B (gas reliability) and C (power economics) are out of scope.

---

## Problem Statement

Data center developers need large contiguous parcels with appropriate zoning, proximity to fiber routes, cooling water access, and low environmental risk — but evaluating thousands of candidate parcels manually is slow and expensive.

This project builds a multi-criteria scoring model that:
1. Ingests public land parcel, water infrastructure, and fiber route data
2. Scores each parcel across four dimensions: size, water access, flood risk, and infrastructure
3. Applies robustness testing via Monte Carlo simulation to identify sites whose rankings are stable under different weight assumptions
4. Outputs a ranked shortlist of 20 high-confidence candidate sites with per-dimension scores

---

## Methodology

### Data Sources (all public)
| Layer | Source | Records |
|-------|--------|---------|
| Parcels | Travis County ArcGIS REST API | ~186,000 |
| Waterbodies | USGS National Map 3DHP FeatureServer | bbox query |
| Flowlines | USGS National Map 3DHP FeatureServer | bbox query |
| Fiber Routes | Public ArcGIS FeatureServer | bbox query |

**Study area:** Travis County, TX (bbox: -98.25, 30.00, -97.30, 30.75)

### Pipeline

```
186,000 parcels (Travis County)
    ↓ Hard filters: ≥50 acres, no water body intersection,
      <5 km to waterbody, <2 km to flowline
630 qualified parcels (0.17%)
    ↓ Re-normalize scores on qualified set
    ↓ Add flood-risk proxy layer
    ↓ K-means geographic clustering (k=20)
    ↓ Monte Carlo: 1,000 simulations with Dirichlet weight sampling
Top 20 recommended sites
```

### Scoring Dimensions
| Feature | Description | Weight range |
|---------|-------------|-------------|
| Acreage | Larger parcels score higher | ~40–50% |
| Water proximity | Closer to waterbody/flowline = better cooling access | ~22–30% |
| Flowline proximity | Secondary water source indicator | ~13–20% |
| Flood risk proxy | Composite of water/flowline proximity + intersection | penalty |

### Monte Carlo Robustness
Rather than a single fixed weighting, we run 1,000 simulations drawing weights from a symmetric Dirichlet(2) distribution. This produces:
- **p_top20**: probability a parcel ranks in top 20 across all weight scenarios
- **confidence_score**: composite index combining p_top20, score stability, and flood safety
- **robust_rank**: final recommendation rank

Only sites that are consistently competitive across diverse weighting assumptions are shortlisted.

---

## Results

**Top 3 Recommended Sites:**

| Rank | Parcel ID | Acres | p_top20 | Confidence |
|------|-----------|-------|---------|------------|
| 1 | 804168 | 118.4 | 0.993 | 0.746 |
| 2 | 324445 | 83.5 | 0.963 | 0.733 |
| 3 | 214268 | 75.2 | 0.935 | 0.715 |

Full results: [`outputs/top20_robust_v3.csv`](outputs/top20_robust_v3.csv)  
Interactive map: [`outputs/map_robust_top20_v3.html`](outputs/map_robust_top20_v3.html)

---

## Repository Structure

```
submission_bundle/
├── notebooks/
│   ├── 01_data_pipeline.ipynb        # Data download, feature extraction
│   └── 02_robustness_analysis.ipynb  # Monte Carlo analysis, final rankings
├── src/
│   ├── build_features.py    # Data download + feature engineering
│   ├── score_models.py      # Scoring model definitions (v1–v4, flood proxy)
│   ├── run_monte_carlo.py   # Monte Carlo simulation engine
│   ├── run_experiments.py   # End-to-end pipeline runner
│   └── generate_outputs.py  # CSV, chart, and map generation
├── outputs/
│   ├── top20_robust_v3.csv          # FINAL ranked output
│   ├── map_robust_top20_v3.html     # Interactive Folium map
│   ├── robustness_v3.csv            # Full metrics for 630 sites
│   └── charts/                      # Visualizations
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Step 1: Download data and build features (~30 min, requires internet)
python src/build_features.py

# Step 2: Run full experiment pipeline
python src/run_experiments.py

# Or: run notebooks interactively
jupyter lab
```

> **Note:** Raw data (~500 MB) is not committed to this repo. Notebooks download it automatically from public APIs on first run. Cached data in `data/raw/` is reused on subsequent runs.

---

## Key Design Decisions

**Why Monte Carlo instead of a fixed weight model?**  
No single weighting of acreage vs. water access is objectively correct — different developers have different priorities. Monte Carlo over a Dirichlet distribution tests which sites perform well under *any reasonable* weighting, not just one assumed preference.

**Why hard filter first?**  
Applying minimum viability thresholds (50 acres, no flood intersection) before scoring prevents the Monte Carlo from wasting simulations on sites that would never pass basic due diligence. It also re-normalizes scores within the competitive range.

**Why flowline proximity separately from waterbody?**  
Waterbodies indicate cooling water *availability*; flowlines indicate water *access routes* and drainage. Sites near flowlines but not directly intersecting them have the best cooling infrastructure without the flood exposure.

---

## Limitations & Future Work

- **Flood risk** is approximated from water proximity — should be replaced with FEMA 100-year flood zone data
- **Fiber not used in final scoring** — downloaded but excluded because fiber coverage was near-uniform across Travis County; needs a wider study area where fiber scarcity is a real constraint
- **No gas infrastructure** — Sub-problem B (PHMSA pipeline reliability) would add the natural gas supply layer
- **No zoning or ownership** — Actual lease feasibility requires county zoning GIS and deed record analysis
- **Single county** — Expanding to all of Texas and Southwest (AZ/NM) would require the same pipeline at larger scale

---

## Dependencies

See [`requirements.txt`](requirements.txt). Core stack: `geopandas`, `pandas`, `numpy`, `scikit-learn`, `folium`, `matplotlib`.
