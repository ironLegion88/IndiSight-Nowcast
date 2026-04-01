# IndiSight-Nowcast: System Setup Guide

This guide provides step-by-step instructions to set up the IndiSight-Nowcast project on a fresh machine (Windows, macOS, or Linux). It covers both standard hardware (CPU-only) setups and high-end workstations (NVIDIA GPUs).

## Phase 1: System Prerequisites

Before touching the code, ensure the following system-level dependencies are installed:

1. **Git:** [Download Git](https://git-scm.com/downloads)
2. **Docker Desktop:** Required for the local PostGIS and Qdrant databases.
   * [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
   * *Windows Users:* Ensure WSL2 (Windows Subsystem for Linux) is enabled in Docker settings.
3. **Pixi:** The package manager used for the project.
   * *Windows (PowerShell):* `irm -useb https://pixi.sh/install.ps1 | iex`
   * *Linux/macOS (Bash):* `curl -fsSL https://pixi.sh/install.sh | sh`
   * *Note: Restart your terminal after installing Pixi.*

---

## Phase 2: Clone & Initialize Environment

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ironLegion88/IndiSight-Nowcast.git
   cd IndiSight-Nowcast
   ```

2. **Install the Python Environment (Pixi):**
   Pixi handles all Python versions, PyTorch binaries, and geospatial libraries automatically.

   * **For Standard Laptops / Member B (CPU Only):**
     ```bash
     pixi install
     ```
   
   * **For the ML Workstation / Member A (NVIDIA GPUs):**
     ```bash
     pixi install -e gpu
     ```

---

## Phase 3: Infrastructure (Databases)

We will use Docker Compose to run PostgreSQL (with PostGIS) and Qdrant locally.

1. Ensure **Docker Desktop** is open and running in the background.
2. Start the databases using the predefined Pixi task:
   ```bash
   pixi run start-db
   ```
   *(To stop them later, you can run `pixi run stop-db`)*

3. **Verify Databases are Running:**
   * PostGIS: `localhost:5432`
   * Qdrant Dashboard: Open `http://localhost:6333/dashboard` in your browser.

---

## Phase 4: Environment Variables (.env)

The project requires specific API keys for the LLM Agent and Google Earth Engine. 

1. Create a file named `.env` in the root directory:
   ```bash
   touch .env
   ```
2. Add the following configurations to the `.env` file:
   ```env
   # Database Configurations (Matches docker-compose.yml)
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_DB=indisight_db
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432

   QDRANT_HOST=localhost
   QDRANT_PORT=6333

   # LLM API Keys (For Member B)
   # Get a free key from Google AI Studio
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Groq API Key (Optional, for blazing fast Llama-3 inference)
   GROQ_API_KEY=your_groq_api_key_here
   ```

---

## Phase 5: Verification & Workflows

To ensure everything is installed correctly, run the following tests:

**1. Test the UI (Streamlit):**
```bash
pixi run ui
```
*This should open a blank or placeholder Streamlit app in your browser at `localhost:8501`.*

**2. Authenticate Google Earth Engine (Member A Only):**
If you are running the satellite extraction scripts, you must authenticate with Google:
```bash
pixi run -e gpu earthengine authenticate
```

---

## Git Branching Reminder
Never work directly on the `main` branch! 
* **Data/ML Tasks (Member A):** `git checkout feat/vision-pipeline`
* **App/DB Tasks (Member B):** `git checkout feat/data-ingestion`