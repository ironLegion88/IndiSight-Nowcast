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
