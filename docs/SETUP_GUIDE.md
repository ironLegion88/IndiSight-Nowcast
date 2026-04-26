# IndiSight-Nowcast: System Setup Guide

This guide provides step-by-step instructions to set up the IndiSight-Nowcast project. It is specifically tailored for **Member A (ML/Vision Workstation)** and **Member B (App/Agent Node)**.

## Phase 1: System Prerequisites

Ensure the following system-level dependencies are installed:

1. **Git:** [Download Git](https://git-scm.com/downloads)
2. **Docker Desktop:** Required for PostGIS and Qdrant.
   - [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - *Windows Users:* Ensure WSL2 is enabled in Docker settings.
3. **Pixi:** The package manager used for the project.
   - *Windows (PowerShell):* `irm -useb https://pixi.sh/install.ps1 | iex`
   - *Linux/macOS (Bash):* `curl -fsSL https://pixi.sh/install.sh | sh`

---

## Phase 2: Clone & Initialize Environment

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ironLegion88/IndiSight-Nowcast.git
   cd IndiSight-Nowcast
   ```

2. **Install the Environment:**
   - **Member A (NVIDIA GPU Workstation):**
     ```bash
     pixi install -e gpu
     ```
   - **Member B (Standard Laptop):**
     ```bash
     pixi install
     ```

---

## Phase 3: Infrastructure & Environment

1. **Start Databases:**
   ```bash
   pixi run start-db
   ```
   *Verify Qdrant at `http://localhost:6333/dashboard`.*

2. **Setup Environment Variables (`.env`):**
   Create a `.env` file in the root with:
   ```env
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_DB=indisight_db
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   GEMINI_API_KEY=your_key
   ```

---

## Phase 4: Data Initialization (Member B Tasks)

Member A needs to run these scripts once to populate the local environment with Member B's data artifacts:

1. **Ingest Tabular Data:**
   ```bash
   pixi run ingest-tabular
   ```
2. **Seed PostGIS Database:**
   ```bash
   pixi run seed-db
   ```
3. **Ingest Policy RAG (Uses GPU if in `-e gpu`):**
   ```bash
   pixi run ingest-rag
   ```
4. **Generate EDA Artifacts:**
   ```bash
   pixi run eda
   ```

---

## Phase 5: Member A Artifact Contract (CRITICAL)

The application node (Member B) expects your ML outputs in a specific "Benchmark Registry" format. To enable the advanced UI features (SHAP, Delta Maps), you **MUST** deliver your results in the following structure:

**Directory Path:** `data/processed/benchmarks/<run_id>/`

### Required Files:
1. **`metrics.json`**:
   ```json
   { "R2": 0.85, "MAE": 1.2, "RMSE": 2.1 }
   ```
2. **`predictions.parquet`**:
   Must contain columns: `district_lgd_code`, `actual`, `predicted`, `delta`.
3. **`shap_summary.json`**:
   A mapping of `district_lgd_code` (as string) to a list of feature importance objects:
   ```json
   {
     "502": [
       {"feature": "Nightlight", "shap_value": 1.5},
       {"feature": "Built-Up Area", "shap_value": -0.3}
     ]
   }
   ```

---

## Phase 6: Verification & Testing

1. **Run the UI:**
   ```bash
   pixi run ui
   ```
2. **Verify Integration:**
   - In the "3D Spatial View" tab, ensure the map renders correctly.
   - If you have placed a valid benchmark run in `data/processed/benchmarks/`, the "Vision Model" sliders will unlock automatically.
   - Check the "AI Policy Assistant" tab and ask: "Which district has the highest electricity coverage?" to test the agent's SQL guardrails.

## Phase 7: Troubleshooting (GPU)

If `torch` does not detect your GPU:
- Run `pixi run python -c "import torch; print(torch.cuda.is_available())"`.
- Ensure you are using the `-e gpu` environment.
- Check that your NVIDIA drivers are up to date (550+ recommended for CUDA 12.1).