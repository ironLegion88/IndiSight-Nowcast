# Architectural & Data Strategy Decisions

## 1. Spatial Joining Standard
**Decision:** All datasets and spatial boundaries will be joined strictly using the **Local Government Directory (LGD) District Code**.
**Reasoning:** Administrative names change, have spelling variations (e.g., "Cuddapah" vs "Kadapa"), and suffer from string-matching errors. LGD codes provide a mathematically guaranteed primary key across NDAP and GeoJSON files.

## 2. Target Variable Selection (Two-Pronged)
**Decision:** 
*   **Phase A (Baseline):** Train models on a curated subset of highly visible infrastructure metrics (Electricity, Clean Fuel, Sanitation, Literacy).
*   **Phase B (Discovery):** Train automated models on all 110+ NFHS variables to discover non-obvious correlations with satellite imagery, systematically narrowing down the best performers.
**Reasoning:** Establishes a scientifically sound baseline while allowing for data-driven discovery.

## 3. The Role of MGNREGA Data
**Decision:** MGNREGA will be utilized in two separate benchmarking pipelines:
1.  **As a Feature:** Fusing historical MGNREGA demand with satellite embeddings to predict NFHS health/wealth outcomes.
2.  **As a Target:** Using satellite imagery to directly "Nowcast" MGNREGA employment demand as a high-frequency proxy for rural economic distress.

## 4. Automated Benchmarking Pipeline
**Decision:** No manual metric tracking. The `model_pipeline` will automatically serialize model metrics (R2, RMSE, MAE), SHAP feature importance arrays, and Plotly visualization JSONs into a `data/processed/benchmarks/` directory. The Streamlit dashboard will dynamically parse this directory to render interactive comparison views.