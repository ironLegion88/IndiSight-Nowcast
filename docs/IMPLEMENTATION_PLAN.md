# IndiSight-Nowcast: Master Implementation Plan & Work Split

## 1. Project Context & Hardware Strategy
This project employs a **Split-Node Architecture** to bypass hardware limitations:
*   **Compute Node (Member A):** High-end Workstation (Ryzen 9, RTX 4070, 96GB RAM). Handles all GPU-bound tasks, heavy remote sensing pipelines, XGBoost training, automated benchmarking, offline PCA, and local LLM (Ollama) hosting.
*   **Application Node (Member B):** Standard hardware. Handles lightweight APIs (Gemini/Groq), database operations (PostGIS/Qdrant), LangChain agent logic, and the Streamlit frontend.

To ensure flawless presentation on a standard laptop, **all heavy dimensionality reductions (PCA), SHAP values, and model predictions will be pre-computed offline** by Member A and serialized into a Benchmark Registry.

---

## 2. Core Architectural Standards
1.  **Primary Key Standard:** All datasets and spatial boundaries MUST be joined using the **Local Government Directory (LGD) District Code**. String-based name matching is strictly prohibited.
2.  **Data Aggregation Standard:** When loading NDAP files (NFHS/MGNREGA), the pipeline MUST filter for `TRU == 'Total'` (or `Residence type == 'Total'`) to prevent double-counting rural and urban populations.
3.  **Automated Benchmarking:** No manual metric tracking. Model runs will automatically dump `metrics.json`, `predictions.parquet`, and `shap_summary.json` into `data/processed/benchmarks/`.

---

## 3. Detailed Module Implementation & Responsibilities

### Phase 1: Data Engineering & EDA
**Goal:** Establish clean, spatially aligned ground-truth data and extract satellite patches.

*   **Task 1.1: Universal Logger & Loader (Member B)**
    *   *File:* `src/utils/logger.py` & `src/data_engine/ndap_loader.py`
    *   *Spec:* Build a cross-platform logger. Build a Pandas loader that automatically cleans headers and applies the `TRU == 'Total'` filter.
*   **Task 1.2: Comprehensive EDA (Member B)**
    *   *File:* `notebooks/01_ndap_eda.ipynb`
    *   *Spec:* Conduct missing value analysis, distribution plotting for baseline metrics, and MGNREGA temporal trend analysis.
*   **Task 1.3: Spatial Boundaries (Member A)**
    *   *File:* `src/data_engine/get_boundaries.py`
    *   *Spec:* Download India District GeoJSON, format column names, and ensure LGD Codes are present.
*   **Task 1.4: Satellite Extraction (Member A)**
    *   *File:* `src/data_engine/extract_gee.py`
    *   *Spec:* Calculate district centroids. Draw 5x5 km bounding boxes. Batch download cloud-free median composites of Sentinel-2 (Optical) and VIIRS (Nightlights). Save as `.tif` in `data/processed/images/`.
*   **Task 1.5: PostGIS Ingestion (Member B)**
    *   *File:* `src/data_engine/ingest_ndap.py`
    *   *Spec:* Create SQLAlchemy tables (`spatial_boundaries`, `ndap_nfhs_metrics`, `ndap_mgnrega_metrics`). Push cleaned DataFrames into the local Dockerized PostgreSQL instance.

### Phase 2: Multi-Modal Machine Learning & Benchmarking
**Goal:** Extract visual features, train XGBoost models, and populate the automated benchmark registry.

*   **Task 2.1: Vision Extraction (Member A)**
    *   *File:* `src/models/vision/vision_extractor.py`
    *   *Spec:* Pass `.tif` images through ResNet-50 (Baseline) and Prithvi-100M/SatMAE (Geospatial). Output `resnet_embeddings.parquet` and `prithvi_embeddings.parquet` mapped to LGD codes.
*   **Task 2.2: Offline PCA Pre-computation (Member A)**
    *   *File:* `src/models/tabular/pca_precompute.py`
    *   *Spec:* Reduce the 2048-d/1024-d embeddings to sizes [64, 128, 256, 512] and save them. This enables instantaneous UI slider switching.
*   **Task 2.3: Automated Benchmark Tracker (Member A)**
    *   *File:* `src/models/tabular/benchmark_tracker.py`
    *   *Spec:* A Python class that serializes experiment runs (R2, MSE, MAE), predictions, and SHAP feature importance arrays to `data/processed/benchmarks/[run_id]/`.
*   **Task 2.4: Model Pipeline & Training (Member A)**
    *   *File:* `src/models/tabular/model_pipeline.py`
    *   *Spec:* Implements the Two-Pronged Strategy:
        *   *Phase A (Baseline):* Train XGBoost on a curated subset (Electricity, Clean Fuel, Sanitation, Literacy).
        *   *Phase B (Discovery):* Loop through all 110+ NFHS metrics automatically to find the most predictable variables.
        *   *MGNREGA Benchmarks:* Train models using MGNREGA as a *feature*, and a separate model predicting MGNREGA as a *target*.

### Phase 3: Agentic AI & RAG
**Goal:** Build the LLM co-pilot capable of SQL queries and policy retrieval.

*   **Task 3.1: RAG Ingestion (Member B)**
    *   *File:* `src/agent/rag_ingest.py`
    *   *Spec:* Parse NITI Aayog PDF reports. Chunk text, generate embeddings using a lightweight SentenceTransformer, and store in Qdrant Vector DB.
*   **Task 3.2: LLM Agent Architecture (Member B)**
    *   *File:* `src/agent/llm_agent.py`
    *   *Spec:* Implement a LangChain ReAct agent using `LiteLLM` (defaults to Gemini 1.5 Flash for dev, Ollama for production). 
    *   *Tools:* Give the agent a `PostGIS_Text2SQL` tool, a `Qdrant_VectorSearch` tool, and a `Fetch_Prediction` tool.

### Phase 4: Streamlit UI & Presentation
**Goal:** Build the interactive dashboard integrating all layers.

*   **Task 4.1: Dashboard UI (Member B & Member A)**
    *   *File:* `app/main.py`
    *   *Spec:* 
        *   **Sidebar:** Dropdowns to read the `data/processed/benchmarks/` directory dynamically (Model: ResNet/Prithvi, PCA: 64-512, Target: NFHS/MGNREGA).
        *   **Map:** Use `pydeck` to render 3D district polygons. Colorize based on prediction vs actual delta.
        *   **Deep-Dive:** District click triggers SHAP waterfall plot rendering.
        *   **Chat:** Interface for interacting with the LangChain Agent.

---

## 4. Weekly Execution Sprint Timeline

### Week 1: Infrastructure & Data Acquisition
*   **Member B (`feat/data-ingestion`):** Setup Logger, build `ndap_loader.py`, run Comprehensive EDA Notebook. Define PostGIS schema and ingest NDAP CSVs and boundaries into Docker database.
*   **Member A (`feat/vision-pipeline`):** Authenticate with Google Earth Engine. Write and execute `extract_gee.py` to download all 5x5km district image patches.

### Week 2: Embeddings & DB Seeding (Integration Point 1)
*   **Member A (`feat/vision-pipeline`):** Write PyTorch `vision_extractor.py`. Extract all raw embeddings. Run `pca_precompute.py`. Push the resulting `.parquet` files to the shared repo/drive.
*   **Member B (`feat/data-ingestion`):** Begin `rag_ingest.py`. Load government PDFs into Qdrant. Verify PostGIS is successfully queried via standard SQL.

### Week 3: Machine Learning & LLM Agent Logic
*   **Member A (`feat/vision-pipeline`):** Build `benchmark_tracker.py` and `model_pipeline.py`. Run Phase A and Phase B automated training loops. Generate the final SHAP and Metric JSONs.
*   **Member B (`feat/agent-logic`):** Build `llm_agent.py`. Construct the Text-to-SQL tool and test it extensively against the PostGIS database. Ensure the agent can accurately answer questions like "Which district has the lowest sanitation score?"

### Week 4: Streamlit UI Integration & Polish
*   **Joint Effort (`main` / `feat/ui-dashboard`):** 
    *   Member B builds the UI skeleton and Agent chat interface.
    *   Member A wires the UI sliders directly to the pre-computed Benchmark Registry JSON/Parquet files.
    *   Implement `pydeck` mapping.
    *   Test the system fully offline (Ollama + pre-computed models) on Member A's laptop to guarantee presentation stability.