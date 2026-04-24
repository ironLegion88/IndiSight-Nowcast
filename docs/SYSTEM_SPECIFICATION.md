# IndiSight-Nowcast: System Specification Document

## 1. Executive Summary
**IndiSight-Nowcast** is a multi-modal Machine Learning and Agentic AI framework designed to solve the "data-lag" in Indian governance. By fusing spatially aligned satellite imagery (Google Earth Engine) with historical tabular data (NITI Aayog NDAP), the system "nowcasts" current district-level socioeconomic and infrastructure health. The platform is augmented by an LLM-driven agent capable of reading the data, executing SQL queries, and suggesting policy interventions using Retrieval-Augmented Generation (RAG).

## 2. Hardware & Deployment Strategy
Due to hardware constraints across the team, the system utilizes a split-node architecture:
*   **Compute Node (Member A):** A high-end workstation (Ryzen 9, RTX 4070, 96GB RAM). Responsible for heavy PyTorch vision extraction, Earth Engine batch downloads, XGBoost model training, offline PCA computations, and local LLM hosting (Ollama).
*   **Application Node (Member B):** Standard hardware. Responsible for API interactions, PostGIS/Qdrant database operations, LangChain agent logic, and Streamlit UI rendering.
*   **Integration:** The Compute Node passes lightweight mathematical artifacts (`.parquet` embeddings, `.json` benchmarks) to the Application Node for seamless, hardware-agnostic presentations.

---

## 3. High-Level System Architecture
The system is divided into four strictly isolated layers:
1.  **Data & Storage Layer:** PostGIS (Tabular/Spatial) + Qdrant (Vector).
2.  **Vision & Embeddings Layer:** PyTorch-based satellite feature extractors.
3.  **Predictive ML Layer:** XGBoost + SHAP automated benchmarking pipeline.
4.  **Agentic AI & Presentation Layer:** LangChain + Streamlit UI.

---

## 4. Detailed Module Specifications

### Module 1: Data Engineering & ETL (`src/data_engine`)
**Function:** Extracts, cleans, and standardizes raw data into the PostGIS database, ensuring spatial alignment.
*   **`get_boundaries.py`:** Fetches official India District GeoJSON polygons. Normalizes district names and assigns Local Government Directory (LGD) Codes.
*   **`extract_gee.py`:** Interfaces with Google Earth Engine (GEE).
    *   *Logic:* Calculates the centroid of each district polygon, draws a 5km x 5km bounding box, and downloads a cloud-free annual median composite of Sentinel-2 (Optical) and VIIRS (Nightlight) imagery.
    *   *Output:* High-resolution `.tif` files saved to `data/processed/images/`.
*   **`ndap_loader.py` & `ingest_ndap.py`:** 
    *   *Logic:* Parses NFHS-4, NFHS-5, and MGNREGA datasets. Strictly filters for `Residence type == 'Total'` to prevent double-counting. Maps datasets to LGD Codes.
    *   *Interaction:* Pushes cleaned DataFrames into the local Dockerized PostGIS database using `SQLAlchemy` and `GeoAlchemy2`.

### Module 2: Vision Extraction Pipeline (`src/models/vision`)
**Function:** Converts raw satellite pixels into dense mathematical vectors (embeddings) representing physical infrastructure.
*   **`vision_extractor.py`:**
    *   *Logic:* Loads PyTorch vision models. Supports a baseline model (`ResNet-50`) and a geospatial-specific foundation model (`Prithvi-100M` or `SatMAE`). 
    *   *Process:* Iterates over the `.tif` images, applies necessary normalizations (handling 4-band to 3-band conversions), and extracts the penultimate layer activations (e.g., 2048-dimensional vectors).
    *   *Output:* Generates `resnet_embeddings.parquet` and `prithvi_embeddings.parquet` containing `(LGD_Code, Embedding_Vector)`.

### Module 3: Tabular ML & Automated Benchmarking (`src/models/tabular`)
**Function:** Learns the statistical mapping between physical satellite embeddings and official NDAP socioeconomic metrics.
*   **`pca_precompute.py`:** 
    *   *Logic:* Takes the high-dimensional embeddings and computes Principal Component Analysis (PCA) reductions (e.g., 64, 128, 256, 512 dimensions). Saves these to allow rapid, real-time UI switching during the presentation.
*   **`model_pipeline.py` & `benchmark_tracker.py`:**
    *   *Interaction:* Fetches embeddings (from Qdrant/Parquet) and NDAP metrics (from PostGIS).
    *   *Logic:* Conducts a two-pronged experimental design:
        *   *Phase A:* Trains models on targeted infrastructure metrics (Electricity, Clean Fuel, Sanitation).
        *   *Phase B:* Automates training across all 110+ NFHS variables to discover non-obvious correlations.
    *   *Algorithm:* XGBoost Regressor.
    *   *Explainability:* Generates SHAP (SHapley Additive exPlanations) values to explain which satellite visual features drove specific predictions.
    *   *Output:* Automatically serializes `metrics.json`, `predictions.parquet`, and `shap_summary.json` into a `data/processed/benchmarks/` registry.

### Module 4: Agentic LLM & RAG (`src/agent`)
**Function:** Acts as the analytical "Brain" of the system, interpreting data and suggesting policy.
*   **`rag_ingest.py`:** Chunks NITI Aayog policy PDF documents using LangChain document loaders, embeds them using an open-source sentence transformer, and loads them into the Qdrant vector database.
*   **`llm_agent.py`:** 
    *   *Logic:* Implements a ReAct (Reasoning + Acting) Agent using `LiteLLM` (allowing seamless switching between Local Ollama and Cloud Gemini APIs).
    *   *Tools Provided to Agent:*
        1.  `Text-to-SQL`: Given a natural language question ("What was the MGNREGA demand in Anantapur in 2019?"), the agent writes a PostgreSQL query, executes it against the PostGIS database, and reads the result.
        2.  `Policy-Search`: Executes a semantic vector search against Qdrant to retrieve policy recommendations.
        3.  `Prediction-Fetch`: Retrieves the XGBoost "Nowcasted" values for a given district.

### Module 5: Presentation Layer (`app/main.py`)
**Function:** The interactive Streamlit dashboard for users and grading evaluators.
*   **UI Components:**
    1.  **Benchmarking Sidebar:** Sliders and dropdowns to select the Vision Model (`ResNet` vs `Prithvi`), PCA Dimensionality (`64` to `512`), and Target Metric.
    2.  **Spatial 3D Map (`pydeck`):** Renders India's districts. Color-codes districts based on the *Delta* between the historical NDAP value and the Nowcasted value.
    3.  **District Deep-Dive:** When a user clicks a district on the map, it displays the SHAP waterfall plot explaining the prediction.
    4.  **Agent Co-Pilot Chat:** A chat window allowing the user to converse with the LLM Agent.

---

## 5. Database Schemas

### PostgreSQL (PostGIS) Relational Schema
*   **Table: `spatial_boundaries`**
    *   `lgd_code` (Integer, Primary Key)
    *   `state_name` (String)
    *   `district_name` (String)
    *   `geometry` (Geometry/Polygon)
*   **Table: `ndap_nfhs_metrics`**
    *   `lgd_code` (Integer, Foreign Key)
    *   `year_code` (Integer)
    *   `metric_name` (String)
    *   `value` (Float)
*   **Table: `ndap_mgnrega_metrics`**
    *   `lgd_code` (Integer, Foreign Key)
    *   `year_code` (Integer)
    *   `households_demanded_work` (Integer)
    *   `person_days_worked` (Integer)

### Qdrant Vector Database Collections
*   **Collection: `policy_documents`**
    *   *Vector:* 768-d (Sentence Transformer embedding of text chunk).
    *   *Payload:* `{ "source_doc": "NITI_Aayog_Health_2022.pdf", "text": "..." }`
*   **Collection: `satellite_embeddings`** (Used if not loading from Parquet)
    *   *Vector:* Variable dimensionality (from Vision Model).
    *   *Payload:* `{ "lgd_code": 502, "model": "resnet", "pca_dim": 128 }`

---

## 6. System Interaction Flow (Step-by-Step Trace)

**Scenario:** User opens the dashboard and investigates "Anantapur" district's economic health.
1.  **Initialization:** Streamlit loads `predictions.parquet` from the Benchmark Registry and renders the `pydeck` map.
2.  **Benchmarking Toggle:** User changes the slider from "ResNet (128-d)" to "Prithvi (512-d)". Streamlit hot-swaps the underlying metric files and instantaneously updates the map colors.
3.  **District Selection:** User clicks "Anantapur".
4.  **Local Fetch:** Streamlit queries PostGIS for the true 2019 NFHS and MGNREGA values and displays them alongside the 2024 Nowcasted prediction.
5.  **Explainability:** The SHAP JSON is parsed, rendering a plot showing that *High Built-Up Area (Feature 42)* and *Low MGNREGA Demand (Feature Tab-3)* pushed the wealth prediction higher.
6.  **Agentic Query:** User types in chat: *"Why is the sanitation score lagging, and what are the government guidelines?"*
7.  **Agent Execution:**
    *   Agent uses `Text-to-SQL` to query PostGIS for historical sanitation trends in Anantapur.
    *   Agent uses `Policy-Search` to query Qdrant for NITI Aayog sanitation frameworks.
    *   Agent synthesizes both data sources and outputs a grounded, factual response.

---

## 7. Evaluation & Success Metrics
The project will be evaluated on the following rigorous MLOps criteria:
1.  **Machine Learning Metrics:** R-squared (R2), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) of the XGBoost predictions against hold-out NDAP tabular data.
2.  **Ablation Studies:** Documenting how prediction accuracy changes when varying the Vision Model (Baseline vs Geospatial) and the Embedding Dimensionality (64 vs 128 vs 512).
3.  **Agentic Reliability:** Validating that the LLM's Text-to-SQL tool generates syntactically correct PostGIS queries without hallucinating nonexistent tables.
4.  **System Latency:** Ensuring the UI remains responsive (< 2 seconds) by strictly utilizing pre-computed benchmarks and offline PCA during the demonstration.