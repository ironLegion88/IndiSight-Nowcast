# IndiSight-Nowcast — Tutorial & Project Blog

## Overview
This short tutorial explains the motivation, architecture, and how to use the current IndiSight-Nowcast codebase and dashboard. It documents the implementation state (April 2026) and provides a concise how-to for reproducing key artifacts locally.

---

## Problem Statement
Satellite imagery availability is intermittent. IndiSight-Nowcast fills temporal gaps by learning spatio-temporal mappings from multispectral Sentinel-2 data to district-level socio-economic indicators (NFHS/NDAP), enabling near-real-time nowcasts at district granularity.

## What the repo currently contains (summary)
- A production-ready data engineering pipeline that harmonizes NFHS/PMGSY/MGNREGA data and seeds PostGIS.
- An EDA pipeline producing 50+ Plotly artifacts and diagnostic CSVs (missingness, Moran's I, outliers).
- An agentic LangChain layer and RAG ingestion scaffold for policy documents (Qdrant integration).
- Vision model code (`Prithvi.py`) and inference scripts (`inference_nowcast.py`, `final_validate.py`, `visualise_comparison.py`) are present. The pretrained model weights `Prithvi_100M.pt` are included at the repository root so you can run end-to-end inference locally (see below).
- A Streamlit dashboard (`app/main.py`) with map, agent, and EDA tabs.

## Quick local run (assumes dependencies are installed via `pixi` or pip)
1. Start PostGIS + Qdrant (docker-compose):

```bash
docker-compose up -d
```

2. Seed PostGIS and prepare data (ensure required env vars in `.env`):

```bash
pixi run python -m src.data_engine.ingest_tabular
pixi run python -m src.data_engine.seed_postgis
```

3. Generate or place EDA artifacts in `data/processed/eda_artifacts/` and ensure spatial file `data/processed/spatial/india_districts_lgd.geojson` exists.

4. Run the dashboard locally:

```bash
pixi run streamlit run app/main.py
```

Notes: The vision inference scripts require Earth Engine authentication. The pretrained weights `Prithvi_100M.pt` are already present at the repository root, so after authenticating Earth Engine you can run inference locally. Example commands:

```bash
# Authenticate Earth Engine first (one-time interactive step):
earthengine authenticate

# Run the nowcast inference using the included weights
python inference_nowcast.py

# Convert latent Z-score outputs to physical reflectance units
python final_validate.py

# Produce the NIR band comparison plot
python visualise_comparison.py
```

These will generate `nowcast_prediction_output.tif`, `IndiSight_Nowcast_PHYSICAL.tif`, and `spectral_comparison_analysis.png` in the repository root.

---

## How to use the Dashboard (current features)
- **3D Spatial View:** Select a metric and year; view district-level values with extrusion. If benchmark artifacts exist, toggle to see prediction deltas and SHAP explainability for districts.
- **AI Policy Assistant:** Agent scaffold is initialized; populate `data/raw/policy_docs/` and run ingestion to test RAG searches.
- **Metrics & EDA:** Loads precomputed Plotly artifacts from `data/processed/eda_artifacts/` for deeper inspection.

---

## EDA Insights (brief)
- The dataset contains ~728 districts with 7 years of NFHS-derived metrics. Spatial autocorrelation (Moran's I) indicates non-random geographic clustering for many health and infrastructure indicators.
- Southern states typically show higher electrification and sanitation metrics compared with some northern and northeastern clusters. Outliers and missingness diagnostics are available under `data/processed/eda_artifacts/`.

---

## EDA Artifact Interpretations
Below are short narrative explanations for key EDA artifacts exported to `docs/figures/` (PNG). Use these paragraphs directly in reports or slide notes.

- **Correlation heatmap (`correlation_heatmap.png`)**: Shows pairwise Pearson correlations between the focal metrics. Strong positive correlations (warm colors) indicate metrics that co-vary spatially—useful to identify redundant predictors. Negative correlations highlight potential trade-offs (e.g., high infrastructure but low health outcomes in outliers).

- **Electrification spatial map (`map_hh_electricity.png`)**: Choropleth of district-level electrification (2019). The map highlights regional clusters of high electrification in southern states and pockets of low coverage in specific northern districts, suggesting targeted policy interventions.

- **Electrification distribution (`distribution_hh_electricity.png`)**: Kernel density / histogram view of district electrification values for 2019. This visual shows the central tendency and tail behavior—important for choosing robust summary statistics.

- **Electrification drift box (`drift_box_hh_electricity.png`)**: Compares distribution of electrification between 2015 and 2019, exposing how many districts made meaningful gains and which remained stagnant.

- **National trend (`trend_hh_electricity.png`)**: Time series of aggregate electrification by year. Useful to contextualize district-level variation against national-level progress.

- **State ranking (`ranking_hh_electricity.png`)**: Top/bottom state ranking for electrification (2019). Handy for executive summaries and spot-checking state-level performance.

- **Road length vs Electrification scatter (`scatter_pmgsy_vs_electricity.png`)**: Scatter plot relating PMGSY road length to district electrification. Helps assess whether infrastructure investment correlates with electrification outcomes at the district level.

## Writing the Report and Next Steps
This repository already contains the raw artifacts needed to draft a reproducible report. Recommended steps:
1. Load the Plotly JSON artifacts from `data/processed/eda_artifacts/` and export PNGs for the static report.
2. Use `eda_dataset_summary.csv` and `morans_i_spatial_stats.csv` to write narrative paragraphs on data quality and spatial clustering.
3. If Member A produces embeddings and benchmarks, validate them with `src/utils/contract_validator.py` and then include SHAP-driven explanations in the report.

---

## Architecture (short)
- Data: PostGIS for geo-tabular, Qdrant for vectors.
- Vision: 3D Tubelet ViT (`Prithvi.py`) for temporal patches.
- Predictive: XGBoost pipeline (not yet trained in repo) producing `predictions.parquet` + `shap_summary.json`.
- Presentation: Streamlit + PyDeck + Plotly.

---

## Contact and Contribution
For questions or contributions, open an issue or submit a PR. Suggested next tasks: generate synthetic embeddings for dashboard demos, write EDA narrative text for each artifact, and prepare a Streamlit Cloud deployment configuration.
