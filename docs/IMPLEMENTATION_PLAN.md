# IndiSight-Nowcast: Implementation Plan & Work Split

## 1. Project Context

*   **Member A:** Handles all GPU-bound tasks, heavy image processing, and local LLM hosting.
*   **Member B:** Handles lightweight APIs, database operations, LangChain logic, and the Streamlit frontend.

To ensure a flawless presentation on a standard laptop, **all heavy dimensionality reductions (PCA) and vision embeddings will be pre-computed offline** and stored as lightweight `.parquet` artifacts.

---

## 2. Non-Blocking Work Split

### Member A:
*Focus: GPU compute, Remote Sensing, Tabular ML, and Offline Fallbacks.*
1. **Satellite Data Pipeline (`src/data_engine/extract_gee.py`):** Write Google Earth Engine scripts to batch-download masked Sentinel-2 and VIIRS imagery for Indian districts.
2. **Vision Embeddings (`src/models/vision_extractor.py`):** Pass raw images through ResNet-50 and Prithvi-100M using PyTorch. Output `[model]_embeddings.parquet`.
3. **Offline PCA Fallbacks (`src/models/pca_precompute.py`):** To support UI sliders during the laptop demo, pre-calculate PCA reductions (e.g., 64, 128, 256, 512 dimensions) for both models and save them.
4. **Predictive Modeling (`src/models/tabular_predictor.py`):** Build the XGBoost training pipeline that maps embeddings to NDAP target metrics. Generate SHAP value arrays.
5. **Local LLM Testing:** Run `Ollama` locally (Llama-3/Mistral) to verify the agent's performance offline.

### Member B:
*Focus: Databases, APIs, Agentic Logic, and Frontend.*
1. **NDAP Data Ingestion (`src/data_engine/ingest_ndap.py`):** Pull NFHS-5 and MGNREGA tabular data + GeoJSON boundaries. Clean and push to the local PostGIS database.
2. **Infrastructure & RAG (`src/agent/rag_ingest.py`):** Maintain `docker-compose.yml`. Chunk NITI Aayog policy PDFs and ingest them into the Qdrant Vector Database.
3. **LLM Agent (`src/agent/llm_agent.py`):** Build the LangChain ReAct agent using `LiteLLM`. Provide it with tools for PostGIS Text-to-SQL and Qdrant Vector Search. Default to the free Gemini 1.5 Flash API for development.
4. **Streamlit UI (`app/main.py`):** Build the dashboard. Integrate `pydeck` for the 3D map, create the chat interface, and wire up Member A's pre-computed PCA artifacts to the UI sliders.

---

## 3. Weekly Execution Timeline

*   **Week 1: Infrastructure & Data:** Member A gets GEE images; Member B gets PostGIS running with NDAP data.
*   **Week 2: The Hand-off:** Member A extracts PyTorch embeddings and sends the `.parquet` files to Member B. Member B loads them into Qdrant.
*   **Week 3: ML & AI Logic:** Member A builds XGBoost/SHAP. Member B builds the LangChain Agent and RAG pipeline.
*   **Week 4: Integration & UI:** Wire Member A's model outputs and Member B's agent into the Streamlit app. Benchmark and test offline capabilities.