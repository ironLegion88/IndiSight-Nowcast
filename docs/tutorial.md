# IndiSight-Nowcast — Practical Guide (April 2026)

Look, the basic idea here is simple: we measure development stuff (sanitation, health, education) like every 5 years through NFHS surveys. Satellites though? They're constantly watching, sending us pictures every few days. We're just trying to connect the dots — if a district *looks* like this from space, with these road investments and labor demand, then its current sanitation/nutrition/schooling is probably around... *this*.

That's the whole thing. Bridge the gap between "we have good pictures" and "we want to know what people's lives look like right now."

If you've only got 20 minutes: run the dashboard, poke the EDA tab, and jump to section 7 below. That's the real story.

---

## What's in the Box

**Data side:** ingestion pipeline that stitches together NFHS + roads + employment data. Spits out EDA plots (50+ Plotly artifacts), spatial tables in PostGIS, CSV diagnostics.

**Dashboard:** Streamlit app. 3D map view (district extrusions), interactive EDA browser, AI agent scaffold (for policy docs), and this tutorial tab.

**Vision side:** Prithvi model code lives here, weights are already in the repo (`Prithvi_100M.pt`), and inference scripts are ready to run. You can generate nowcast imagery locally.

**Predictions:** Benchmarking system expects predictions + SHAP explanations to plug in. Contract-driven, so new runs just go into a folder and the dashboard picks them up automatically.

---

## Getting Running (The Path of Least Resistance)

### Install Pixi, then bootstrap everything

```bash
pixi install
```

If you've got a CUDA GPU and want it:

```bash
pixi install -e gpu
```

### Start the databases

```bash
pixi run start-db
```

This starts PostGIS and Qdrant (Docker). Coffee break time — it takes a minute.

### Load the data

```bash
pixi run ingest-tabular
```

This harmonizes NFHS, roads, employment. Writes parquets/CSVs under `data/processed/tabular/`.

```bash
pixi run seed-db
```

Loads spatial stuff into PostGIS. If this fails, check your `.env` (DB credentials, host) and confirm Docker is actually running.

---

## Generate the EDA Artifacts (This is where the insights live)

```bash
pixi run eda
```

You'll get a bunch of Plotly JSON files in `data/processed/eda_artifacts/` plus some CSV diagnostics. This is the raw material — maps, distributions, trends, scatter plots of every outcome vs every program variable.

Export them to PNG so we can embed them in a static report:

```bash
pixi run python scripts/export_eda_pngs.py
```

Outputs land in `docs/figures/`. The static report (`index.html`) already embeds these.

---

## Run the Dashboard

```bash
pixi run ui
```

Point your browser to `http://localhost:8501`. Try the **Metrics & EDA** tab first — pick a map, look at its drift/trend version. Flip between 2015 and 2019 on the **3D Spatial View** tab to see what changed.

---

## Vision Nowcasting (Optional, but Cool)

If you want to play with reconstructing satellite imagery using Prithvi:

```bash
earthengine authenticate
```

(One-time setup. Lets the script pull HLS data.)

Then run the inference pipeline:

```bash
pixi run python inference_nowcast.py
pixi run python final_validate.py
pixi run python visualise_comparison.py
```

You'll get GeoTIFFs and a comparison plot in the repo root. The point here is *does the reconstruction look plausible* — not pixel-perfect truth, but does it capture structure? That's the foundation for using embeddings downstream.

---

## The EDA: What's Actually Happening

Forget the fancy structure — let me just tell you what we learned from the data.

### Geographies matter *a lot*

Most development outcomes cluster spatially. Good news: satellite imagery clusters the same way. That's your signal. Bad news: if you ignore geography in your model, you'll miss half the story. Use spatial embeddings, neighborhood context, regional priors — don't treat districts as independent rows.

Check `morans_i_spatial_stats.csv` if you want the numbers. But honestly, just look at the district maps. You'll see it immediately.

### The "infrastructure bundle" is real

Electricity, sanitation, clean cooking fuel, institutional births, women's education all co-vary. Makes sense — they're all proxies for the same underlying "development infrastructure." But then there's **women's anemia**, which is weirdly independent. It doesn't follow the pattern. That tells you: anemia needs its own levers (supplementation, infection control, diet), not just "build more stuff."

### Ceiling effect = trap for your model

Electrification is basically 95–100% in most districts by 2019. Your model will *love* this — easy to predict! Then someone asks "which lagging districts need urgent intervention?" and your model is useless because it just predicts 98% everywhere. Lesson: evaluate on the tail that matters, not averages.

### Progress isn't linear or universal

Between 2015–2019:
- Infrastructure usually improves
- Child stunting usually improves (gets lower, which is good)
- Women's anemia? **Worsens**. Backward. That one outcome gets worse even while the rest of the country "develops"

That's wild. It tells you development isn't one-dimensional. Some policies help nutrition, some help sanitation, some help nothing. You can't expect one magic model to handle everything.

### PMGSY/MGNREGA: the data's honest about it

We throw road investment and employment demand into the scatter plots vs every outcome. Mostly weak relationships. Here's why:

1. **Vertical bands**: program variables are at state level, outcomes at district level. So all districts in Maharashtra get the same x-axis value. That means it's a *coarse signal*, not a precise district driver.

2. **Targeting is confounding**: roads cost more in remote/mountainous districts. So "higher spend ↔ worse outcome" might just mean "we invest more where it's harder," not "the investment failed." You need to be careful not to misinterpret causality.

The practical bit: *use* these features, but don't rely on them alone. Pair them with satellite embeddings and ideally some district-level covariates.

---

## How It Actually Works (The Mental Model)

Think of three loops:

**Data loop** (you can do this now): ingest → PostGIS → EDA → report/dashboard

**Vision loop** (you can do this now): Prithvi → produce embeddings → radiometric validation

**Prediction loop** (depends on your work): embeddings + tabular features → train model → produce predictions + SHAP → drop into `data/processed/benchmarks/` → dashboard picks it up

The dashboard has a validator (`src/utils/contract_validator.py`) that checks benchmark runs. You just need to follow the contract (predictions.parquet + shap_summary.json) and you're good.

---

## Stuff That Will Bite You (Learned So You Don't Have To)

- **Kaleido breaks on export?** Plotly + Kaleido need to be in the same Pixi env. Re-run `pixi install`.
- **EDA files missing?** Run `pixi run eda` first. The PNG exporter needs those JSON files.
- **Maps are blank?** Make sure `data/processed/spatial/india_districts_lgd.geojson` exists and has the right district codes.
- **Earth Engine auth fails?** Run `earthengine authenticate` and verify your account has HLS access.
- **Windows paths weird?** Use `pixi run ...` to stay in the managed environment. Easier than debugging path nonsense.

---

## What's Next If You Want to Build Real Nowcasts

1. **Pick 2–3 targets.** Sanitation, clean cooking fuel, stunting — strong spatial structure, meaningful drift.
2. **Think about evaluation.** For saturated targets like electrification, use tail-aware metrics. For others, region-stratified validation.
3. **Generate embeddings.** Prithvi per district-year, aligned to NFHS survey years.
4. **Start simple.** Tabular baseline first. Then layer embeddings. See what helps.
5. **Validate the model.** Write predictions + SHAP into a benchmark run folder. The dashboard validator will check it.

Treat this file as living docs — add notes as you change things. Future you (or someone else) will appreciate it.

NFHS-style outcomes (nutrition, anemia, sanitation, schooling) are **high-value** but **low-frequency**: they move on policy timescales, yet we measure them sparsely. Satellite imagery is **high-frequency** and **spatially complete**, but it’s only a proxy.

IndiSight-Nowcast is the bridge:

- Learn a mapping from **satellite-derived representations** (e.g., Prithvi embeddings / spatio-temporal patches) + **program signals** (PMGSY/MGNREGA) → **district-level socio-economic indicators**.
- Use that mapping to produce *inter-survey* estimates: “Given what the district looks like now, what should its development indicators be *now*?”

This repo intentionally keeps those layers separate:
- **Data engineering + EDA** is stable and reproducible.
- **Vision inference** is present and runnable.
- **Downstream predictive benchmarking** (predictions + SHAP) is scaffolded via a contract, but depends on benchmark artifacts.

---

## 1) What You’ll Build / Reproduce

By the end of this tutorial you can:

- Bring up the local services (PostGIS + Qdrant) and run the dashboard.
- Generate EDA artifacts (Plotly JSON + diagnostic CSVs) and export PNGs for static reporting.
- Read the EDA with a “nowcasting lens”: understand spatial clustering, ceiling effects, drift, and why some outcomes are harder.
- (Optional) run Prithvi-based nowcast reconstruction using the included `Prithvi_100M.pt` weights.

---

## 2) Repo Tour (What Exists *As Implemented*)

### 2.1 Data / EDA
- `src/data_engine/ingest_tabular.py`: harmonizes NFHS + PMGSY + MGNREGA and writes processed tables.
- `src/data_engine/seed_postgis.py`: loads geospatial assets into PostGIS.
- `src/data_engine/eda_pipeline.py`: generates EDA artifacts under `data/processed/eda_artifacts/`:
	- Plotly JSON: `map_*.json`, `distribution_*.json`, `trend_*.json`, `drift_box_*.json`, `ranking_*.json`, `scatter_*.json`, plus `correlation_heatmap.json`.
	- CSV diagnostics: missingness, Moran’s I, outliers, quality-by-year.

### 2.2 Dashboard
- `app/main.py`: Streamlit app with tabs:
	- **3D Spatial View**: district extrusions (historical + optional benchmark deltas).
	- **Metrics & EDA**: interactive artifact browser.
	- **AI Policy Assistant**: ingestion scaffold + agent skeleton.
	- **Tutorial**: this document.

### 2.3 Vision / Nowcast Reconstruction
- `Prithvi.py`: the model architecture.
- `Prithvi_100M.pt`: pretrained weights **already present in the repo root**.
- `inference_nowcast.py`, `final_validate.py`, `visualise_comparison.py`: runnable pipeline for producing GeoTIFF outputs and comparisons.

---

## 3) Setup (Pixi + Docker) — The Clean, Reproducible Path

### 3.1 Prerequisites
- Pixi installed
- Docker installed
- (Optional for vision): Google Earth Engine access + authentication

### 3.2 Create the environment

From the repo root:

```bash
pixi install
```

If you have a CUDA GPU and want GPU-enabled Torch (optional):

```bash
pixi install -e gpu
```

### 3.3 Start services

```bash
pixi run start-db
```

This boots PostGIS + Qdrant as defined in `docker-compose.yml`.

### Load the data

```bash
pixi run ingest-tabular
```

This harmonizes NFHS, roads, employment. Writes parquets/CSVs under `data/processed/tabular/`.

```bash
pixi run seed-db
```

Loads spatial stuff into PostGIS. If this fails, check your `.env` (DB credentials, host) and confirm Docker is actually running.

---

## 5) Run the EDA Pipeline (The “Truth Serum”)

### 5.1 Generate EDA artifacts

```bash
pixi run eda
```

You should see artifacts in:
- `data/processed/eda_artifacts/` (Plotly JSON + diagnostic CSVs)

### 5.2 Export PNGs for static reporting

The repo contains a batch exporter that converts *all* Plotly JSON artifacts to PNG:

```bash
pixi run python scripts/export_eda_pngs.py
```

Outputs land in:
- `docs/figures/`

Those PNGs are what the static report consumes.

### 5.3 Open the report

Open `index.html` in your browser. It is designed to be a single-file narrative report embedding the exported PNGs.

---

## Run the Dashboard

```bash
pixi run ui
```

Point your browser to `http://localhost:8501`. Try the **Metrics & EDA** tab first — pick a map, look at its drift/trend version. Flip between 2015 and 2019 on the **3D Spatial View** tab to see what changed.

---

## 7) (Optional) Run the Vision Nowcast Reconstruction (Prithvi)

This part is about reconstructing / nowcasting multispectral imagery (HLS-style) using the Prithvi-100M backbone.
It’s a separate “vision-core” track that can feed embeddings/features into the downstream socio-economic nowcast.

### 7.1 Earth Engine authentication (one-time)

```bash
earthengine authenticate
```

### 7.2 Run inference

Weights are already present (`Prithvi_100M.pt`), so you can run:

```bash
pixi run python inference_nowcast.py
pixi run python final_validate.py
pixi run python visualise_comparison.py
```

Expected outputs in repo root:
- `nowcast_prediction_output.tif`
- `IndiSight_Nowcast_PHYSICAL.tif`
- `spectral_comparison_analysis.png`

Interpretation tip: the point here is *radiometric plausibility* and structural fidelity, not “perfect pixel truth.”
In a full nowcasting system, we’d use the vision model to produce stable representations that correlate with downstream outcomes.

---

## 8) The EDA “Blog”: What We Learn (And How It Changes the Modeling Plan)

This section is intentionally narrative. The goal is not to admire plots; it’s to decide:

- Which outcomes are good targets for nowcasting?
- What failure modes we must anticipate?
- What features are likely to matter?

### 8.1 First principle: Spatial clustering is your friend

If district outcomes are geographically clustered, satellite imagery becomes a powerful proxy because built form, land use,
night-light patterns, and agricultural intensity also cluster.

Where this shows up in artifacts:
- `morans_i_spatial_stats.csv` (high Moran’s I across many metrics)
- the district maps (`map_*.json` → exported PNGs)

Modeling implication:
- A nowcasting model should exploit *spatial structure* (embeddings, neighborhood context, region priors), not treat districts as IID rows.

### 8.2 Correlation heatmap: “infrastructure bundle” vs “health complexity”

The correlation heatmap typically reveals:
- electricity, sanitation, clean cooking fuel, institutional births, women’s schooling co-move (an “infrastructure + services” bundle)
- child stunting tends to move in the opposite direction (higher development ↔ lower stunting)
- women’s anemia is often less tightly coupled than you’d hope

Why that matters:
- It warns against expecting a single latent “development factor” to explain every health outcome.
- It also tells you where multi-task learning might work (bundle outcomes) vs where it might fail (anemia).

### 8.3 Distributions: the ceiling effect trap

Electrification (and sometimes institutional births) has many districts near 95–100%.

That creates a modeling trap:
- A model can look great on average metrics while being useless where policy needs it most (the lagging tail).

Practical fix:
- Evaluate with tail-aware metrics or explicitly focus on low-coverage districts.

### 8.4 Drift & trends: progress is real — but not uniform

Between 2015 and 2019:
- infrastructure indicators often improve (rightward shifts)
- child stunting often improves (downward shift; lower is better)
- women’s anemia is the standout: it **worsens** in the drift/trend artifacts

Interpretation:
- “development progress” is not one-dimensional.
- anemia may require additional predictors (diet, disease environment, supplementation coverage proxies) and should not be treated as a simple byproduct of infrastructure.

### 8.5 Scatter plots (PMGSY/MGNREGA): useful, but not causal

Two patterns to watch:

1) **Macro-vs-micro striping**
If a program variable is at state granularity and outcomes are at district granularity, points align in vertical bands.
That doesn’t make the plot useless — it makes it a *coarse prior*, not a district driver.

2) **Confounding by targeting**
Road spend/cost can be higher in districts that are remote, mountainous, or historically underserved.
So “higher spend ↔ worse outcome” may reflect targeting rather than failure.

Modeling implication:
- Include these program features, but don’t rely on them as sole predictors.
- Combine with satellite embeddings and (ideally) district-level covariates.

---

## 9) How the Pieces Fit (A Practical Mental Model)

Think of IndiSight-Nowcast as three concentric loops:

1) **Data loop (reproducible today)**
Ingest → PostGIS → EDA artifacts → report/dashboard.

2) **Vision loop (runnable today)**
Prithvi inference → radiometric validation → representations.

3) **Benchmark loop (contract-driven)**
Embeddings + tabular → predictions + SHAP → validated benchmark runs.

The dashboard already has a validator (`src/utils/contract_validator.py`) for benchmark runs placed under `data/processed/benchmarks/`.

---

## 10) Common Gotchas (So You Don’t Lose a Day)

- **Kaleido exports fail:** ensure Plotly + Kaleido are installed in the Pixi env. Re-run `pixi install`.
- **EDA artifacts missing:** run `pixi run eda` before exporting PNGs.
- **Maps don’t render:** ensure `data/processed/spatial/india_districts_lgd.geojson` exists and has the expected district codes.
- **Earth Engine errors:** run `earthengine authenticate` and verify your account has access.
- **Windows path weirdness:** prefer `pixi run ...` so you stay inside the managed environment.

---

## 11) What to Do Next (If You’re Extending the Project)

If your goal is “real district nowcasts,” the next meaningful steps are:

1) Decide targets: pick 2–3 outcomes with strong spatial structure and meaningful drift (sanitation, clean cooking fuel, stunting).
2) Define evaluation: tail-aware evaluation for saturated targets; region-stratified validation.
3) Produce embeddings: Prithvi embeddings per district-year (or per time window) aligned to NFHS years.
4) Train a baseline: a simple tabular baseline first, then add embeddings.
5) Populate benchmarks: write `predictions.parquet` + `shap_summary.json` into a benchmark run folder and validate with the contract.

If you want, treat this file as living documentation — add “what we changed” notes as you iterate.
