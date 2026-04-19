"""
build_features.py

Downloads raw geospatial data from public APIs and builds the parcel feature table
used for site scoring. Saves outputs to data/raw/ and data/processed/.

Data sources:
  - Travis County parcels:  ArcGIS REST API (paginated)
  - USGS waterbodies/flowlines: National Map 3DHP FeatureServer
  - Fiber optic routes: public ArcGIS FeatureServer

Usage:
    python src/build_features.py
"""

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BBOX = (-98.25, 30.00, -97.30, 30.75)  # Austin / Travis County, EPSG:4326
TARGET_CRS = "EPSG:3083"               # NAD83 / Texas Centric Albers Equal Area (meters)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PARCEL_URL = "https://taxmaps.traviscountytx.gov/arcgis/rest/services/Parcels/MapServer/0/query"
FLOWLINE_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/3DHP_all/FeatureServer/50/query"
WATERBODY_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/3DHP_all/FeatureServer/60/query"
FIBER_URL = "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Fiber_Optic_Routes/FeatureServer/0/query"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_parcels(out_file: Path, page_size: int = 2000) -> gpd.GeoDataFrame:
    """Download all Travis County parcels via paginated ArcGIS REST query."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    all_gdfs = []
    offset = 0
    batch = 1

    while True:
        params = {
            "where": "1=1",
            "outFields": "*",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        print(f"Downloading parcel batch {batch} (offset={offset})...")
        r = requests.get(PARCEL_URL, params=params, timeout=120)
        r.raise_for_status()
        data = r.json()

        if "features" not in data or len(data["features"]) == 0:
            print("No more parcel features.")
            break

        gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
        all_gdfs.append(gdf)
        print(f"  Got {len(gdf)} parcels")

        if len(gdf) < page_size:
            break
        offset += page_size
        batch += 1

    parcels = pd.concat(all_gdfs, ignore_index=True)
    parcels = gpd.GeoDataFrame(parcels, geometry="geometry", crs="EPSG:4326")
    parcels[["PROP_ID", "geometry"]].to_file(out_file, driver="GeoJSON")
    print(f"Saved {len(parcels)} parcels -> {out_file}")
    return parcels


def download_arcgis_bbox(base_url: str, bbox: tuple, out_file: Path,
                         page_size: int = 2500) -> gpd.GeoDataFrame:
    """Download features from an ArcGIS FeatureServer within a bounding box."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    all_gdfs = []
    offset = 0
    batch = 1

    while True:
        params = {
            "where": "1=1",
            "outFields": "*",
            "f": "geojson",
            "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        print(f"Downloading batch {batch} from {base_url}...")
        r = requests.get(base_url, params=params, timeout=180)
        r.raise_for_status()
        data = r.json()

        if "features" not in data or len(data["features"]) == 0:
            print("No more features.")
            break

        gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
        all_gdfs.append(gdf)
        print(f"  Got {len(gdf)} features")

        if len(gdf) < page_size:
            break
        offset += page_size
        batch += 1

    if not all_gdfs:
        raise ValueError(f"No features downloaded from {base_url}")

    out = pd.concat(all_gdfs, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    out.to_file(out_file, driver="GeoJSON")
    print(f"Saved {len(out)} features -> {out_file}")
    return out


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def minmax_scale(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]. Handles constant series."""
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def build_features(parcels_raw: gpd.GeoDataFrame,
                   fiber_raw: gpd.GeoDataFrame,
                   flow_raw: gpd.GeoDataFrame,
                   waterbodies_raw: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute parcel-level features from raw geodataframes.

    Steps:
      1. Reproject all layers to TARGET_CRS (meters)
      2. Compute parcel area (acres) and centroid coordinates
      3. Spatial nearest-join: parcel centroids to fiber, waterbodies, flowlines
      4. Spatial intersect: flag parcels that overlap a waterbody
      5. Min-max normalize distances and area to [0, 1]

    Returns a flat pandas DataFrame (one row per parcel).
    """
    # 1. Reproject
    parcels_m = parcels_raw[["PROP_ID", "geometry"]].copy().to_crs(TARGET_CRS)
    fiber_m = fiber_raw[["geometry"]].copy().to_crs(TARGET_CRS)
    flow_m = flow_raw[["geometry"]].copy().to_crs(TARGET_CRS)
    wb_m = waterbodies_raw[["geometry"]].copy().to_crs(TARGET_CRS)

    # 2. Parcel geometry attributes
    parcels_m["parcel_id"] = parcels_m["PROP_ID"].astype(str)
    parcels_m["parcel_area_sqm"] = parcels_m.geometry.area
    parcels_m["parcel_area_sqkm"] = parcels_m["parcel_area_sqm"] / 1_000_000
    parcels_m["parcel_area_acres"] = parcels_m["parcel_area_sqm"] / 4046.8564224

    # Centroids: store lat/lon in WGS84 for later use
    centroids_m = parcels_m[["parcel_id", "geometry"]].copy()
    centroids_m["geometry"] = centroids_m.geometry.centroid
    centroids_wgs84 = centroids_m.to_crs("EPSG:4326")
    centroids_m["centroid_lon"] = centroids_wgs84.geometry.x.values
    centroids_m["centroid_lat"] = centroids_wgs84.geometry.y.values

    # 3. Nearest-distance joins (parcel centroid to each layer)
    def nearest_dist(centroids, ref_layer, dist_col, km_col):
        ref = ref_layer[["geometry"]].copy()
        ref["_id"] = range(len(ref))
        joined = gpd.sjoin_nearest(
            centroids[["parcel_id", "geometry"]],
            ref,
            how="left",
            distance_col=dist_col,
        )
        joined[km_col] = joined[dist_col] / 1000.0
        return joined[["parcel_id", dist_col, km_col]].drop_duplicates("parcel_id")

    nearest_fiber = nearest_dist(centroids_m, fiber_m,
                                 "dist_to_fiber_m", "dist_to_fiber_km")
    nearest_water = nearest_dist(centroids_m, wb_m,
                                 "dist_to_waterbody_m", "dist_to_waterbody_km")
    nearest_flow = nearest_dist(centroids_m, flow_m,
                                "dist_to_flowline_m", "dist_to_flowline_km")

    # 4. Waterbody intersection flag (parcel polygon overlaps waterbody polygon)
    wb_intersect = gpd.sjoin(
        parcels_m[["parcel_id", "geometry"]],
        wb_m[["geometry"]],
        how="left",
        predicate="intersects",
    )
    wb_intersect["intersects_waterbody"] = wb_intersect["index_right"].notna().astype(int)
    wb_flag = wb_intersect.groupby("parcel_id", as_index=False)["intersects_waterbody"].max()
    parcels_m = parcels_m.merge(wb_flag, on="parcel_id", how="left")
    parcels_m["intersects_waterbody"] = parcels_m["intersects_waterbody"].fillna(0).astype(int)

    # 5. Assemble and normalize
    base = (
        parcels_m[["parcel_id", "parcel_area_sqm", "parcel_area_sqkm",
                   "parcel_area_acres", "intersects_waterbody"]]
        .drop_duplicates("parcel_id")
        .copy()
    )
    centroids_base = (
        centroids_m[["parcel_id", "centroid_lat", "centroid_lon"]]
        .drop_duplicates("parcel_id")
        .copy()
    )

    feature_df = (
        base
        .merge(centroids_base, on="parcel_id", how="left", validate="one_to_one")
        .merge(nearest_fiber, on="parcel_id", how="left", validate="one_to_one")
        .merge(nearest_water, on="parcel_id", how="left", validate="one_to_one")
        .merge(nearest_flow, on="parcel_id", how="left", validate="one_to_one")
    )

    feature_df["acreage_score"] = minmax_scale(feature_df["parcel_area_acres"])
    feature_df["water_score"] = 1 - minmax_scale(feature_df["dist_to_waterbody_km"])
    feature_df["flowline_score"] = 1 - minmax_scale(feature_df["dist_to_flowline_km"])
    feature_df["water_intersection_penalty"] = feature_df["intersects_waterbody"].astype(float)

    return feature_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    parcel_file = RAW_DIR / "parcels" / "travis_parcels.geojson"
    flowline_file = RAW_DIR / "water" / "flowlines.geojson"
    waterbody_file = RAW_DIR / "water" / "waterbodies.geojson"
    fiber_file = RAW_DIR / "fiber" / "fiber_routes.geojson"

    # Download only if not already cached
    if not parcel_file.exists():
        download_parcels(parcel_file)
    if not flowline_file.exists():
        download_arcgis_bbox(FLOWLINE_URL, BBOX, flowline_file)
    if not waterbody_file.exists():
        download_arcgis_bbox(WATERBODY_URL, BBOX, waterbody_file)
    if not fiber_file.exists():
        download_arcgis_bbox(FIBER_URL, BBOX, fiber_file)

    print("Loading raw data...")
    parcels = gpd.read_file(parcel_file)
    fiber = gpd.read_file(fiber_file)
    flow = gpd.read_file(flowline_file)
    waterbodies = gpd.read_file(waterbody_file)

    print("Building features...")
    feature_df = build_features(parcels, fiber, flow, waterbodies)

    required_cols = [
        "parcel_id", "parcel_area_acres", "intersects_waterbody",
        "centroid_lat", "centroid_lon",
        "dist_to_waterbody_km", "dist_to_flowline_km",
        "acreage_score", "water_score", "flowline_score", "water_intersection_penalty",
    ]
    feature_df_clean = feature_df[required_cols].dropna().copy()

    feature_df_clean.to_parquet(PROCESSED_DIR / "features_clean.parquet", index=False)
    feature_df_clean.to_csv(PROCESSED_DIR / "features_clean.csv", index=False)
    print(f"Saved {len(feature_df_clean)} parcels -> data/processed/features_clean.parquet")


if __name__ == "__main__":
    main()
