# IndiSight-Nowcast: Multi-Modal Socioeconomic Nowcasting

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-green.svg)
![Framework](https://img.shields.io/badge/framework-LangChain%20%2B%20Streamlit-orange)

## Overview
IndiSight-Nowcast addresses the "data-lag" problem in Indian governance. While official surveys (NFHS, MGNREGA) are updated infrequently, satellite imagery provides a real-time signal of economic activity. This project uses **Multi-Modal Machine Learning** to predict current socioeconomic metrics by fusing NDAP tabular data with satellite-derived features.

## Tech Stack
- **Data:** NITI Aayog NDAP (Tabular), Sentinel-2 (Optical), VIIRS (Nightlights).
- **ML/AI:** XGBoost, PyTorch (Vision Transformers), SHAP (Explainability).
- **Databases:** PostGIS (Spatial-Relational), Qdrant (Vector/RAG).
- **Agentic Layer:** LangChain-based SQL + Tool-calling Agent.
- **Frontend:** Streamlit.

## Key Objectives
1. **Bridge the Data Gap:** Nowcast NFHS-5 metrics using weekly satellite revisits.
2. **Explainable AI:** Identify the visual drivers of local economic growth.
3. **Policy Co-Pilot:** An LLM agent that reads maps, queries data, and suggests policy interventions.

## Project Structure
```
IndiSight-Nowcast/
├── .github/               # CI/CD workflows
├── data/                  # Raw and processed data
│   ├── raw/               # NDAP CSVs, GeoJSONs
│   └── processed/         # Cleaned features, satellite embeddings
├── notebooks/             # EDA and model prototyping
│   ├── 01_ndap_eda.ipynb
│   ├── 02_satellite_extraction.ipynb
│   └── 03_model_training.ipynb
├── src/                   # Source code
│   ├── data_engine/       # ETL scripts for NDAP API & GEE
│   ├── models/            # XGBoost, ResNet/ViT architectures
│   ├── agent/             # LangChain agentic logic
│   └── utils/             # Database and spatial helpers
├── database/              # SQL schemas, Vector DB init scripts
├── app/                   # Streamlit dashboard
├── docs/                  # Reports, images, documentation
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignore .env and large datasets
└── README.md              # This file
```

## Data Sources

- **NDAP:** National Data Agro-Meteorology Portal (weather & crop data)
- **GEE:** Google Earth Engine (satellite imagery)
- **Geospatial:** Administrative boundaries, crop zones

## Development Guidelines

To maintain code quality and a clean project history, this repository follows strict Git standards.

### Branching Strategy
Follow a **Conventional Branching** model. Never commit directly to `main`. Create a branch using the following prefixes:
- `feat/` : New features (e.g., `feat/satellite-pipeline`)
- `fix/` : Bug fixes (e.g., `fix/null-values-ndap`)
- `docs/` : Documentation updates
- `refactor/` : Code restructuring without functional changes
- `chore/` : Maintenance tasks (dependencies, config)

### Commit Message Convention
Commits must follow the **Conventional Commits** specification:
`type(scope): description`

**Common Types:**
- `feat`: A new feature for the user
- `fix`: A bug fix
- `docs`: Documentation changes
- `refactor`: Production code change that neither fixes a bug nor adds a feature
- `perf`: Code change that improves performance
- `chore`: Updating grunt tasks etc; no production code change

**Example:**
`feat(llm): add RAG capability for NITI Aayog policy documents`

### Workflow
1. Create a branch: `git checkout -b feat/your-feature-name`
2. Commit changes: `git commit -m "feat(scope): describe your change"`
3. Push and Open a PR to `main`.