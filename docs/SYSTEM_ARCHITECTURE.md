### 1. High-Level System Architecture

We will split the system into four distinct layers. Since you have 96GB RAM and 16 cores, we will run the entire infrastructure locally using **Docker Compose**.

1.  **Data & Storage Layer:**
    *   **Spatial-Relational DB:** PostgreSQL + PostGIS (stores NDAP NFHS-5, MGNREGA tabular data, and India district GeoJSON polygons).
    *   **Vector DB:** Qdrant (stores RAG policy documents and pre-computed satellite image embeddings).
    *   **Image Cache:** Local file system mapping or MinIO (local S3 alternative) to store downloaded Sentinel-2 tiles so you don't hit API rate limits during demos.
2.  **Model & Inference Layer (The Engine):**
    *   **Vision Embedding Extractor:** A PyTorch service running on the RTX 4070 that takes satellite images and outputs 1D feature vectors.
    *   **Multi-Modal Predictor:** XGBoost / LightGBM models trained to map visual embeddings to NDAP tabular metrics.
3.  **Agentic LLM Layer (The Brain):**
    *   **LLM Backend:** LiteLLM proxy routing requests either to Local Ollama or Free Cloud APIs.
    *   **LangChain Agent:** Has "Tools" (Text-to-SQL for PostGIS, Vector-Search for Qdrant, Prediction-Trigger for XGBoost).
4.  **Presentation Layer (The UI):**
    *   **Frontend:** Streamlit with `pydeck` (Deck.gl wrapper for Python) for 3D interactive spatial mapping and model toggles.

---

### 2. Architectural Trade-offs & Decisions

#### A. Vector Database: ChromaDB vs. Qdrant vs. Milvus
*   **ChromaDB:**
    *   *Pros:* Python-native, zero-setup, great for quick prototypes.
    *   *Cons:* In-memory or SQLite-backed. Poor scalability, lacks advanced filtering, feels like a "toy" for an advanced grading rubric.
*   **Milvus:**
    *   *Pros:* Enterprise-grade, massive scalability.
    *   *Cons:* Extremely heavy. Requires Zookeeper/etcd, MinIO, and multiple microservices. Overkill and will clutter your local Docker environment.
*   **Qdrant (Winner):**
    *   *Pros:* Written in Rust (blazing fast, low memory footprint). Runs perfectly as a single Docker container. Native support for complex payload filtering (e.g., "Find embeddings for *District X* where *Year = 2024*").
    *   *Cons:* Requires learning a slightly more complex API than Chroma.
*   **Decision:** **Qdrant**. It provides the professional, production-ready feel your professors will look for without eating up your workstation's resources.

#### B. Vision Models: ResNet-50 vs. ViT vs. Specialized Earth Observation (EO) Models
*   **ResNet-50 (ImageNet pre-trained):**
    *   *Pros:* Extremely fast, minimal VRAM.
    *   *Cons:* ImageNet features (dogs, cars) don't map well to satellite imagery natively.
*   **Vision Transformer (ViT-Base):**
    *   *Pros:* Captures global context better (useful for sprawling road networks).
    *   *Cons:* Data hungry, higher compute. Standard ViT still lacks remote sensing context.
*   **Domain-Specific Foundation Models (Winner):**
    *   *Options:* **Prithvi-100M** (IBM/NASA geospatial foundation model available on HuggingFace) or **ResNet-50-BigEarthNet** (pre-trained specifically on Sentinel-2 data).
    *   *Decision:* Provide a **UI Toggle** between a standard baseline (`ResNet-50`) and a specialized model (`Prithvi-100M` or `SatMAE`). 
    *   *Engineering Trick:* Do NOT run the vision models live on the UI slider. Pre-extract the embeddings for all districts using *both* models offline, store them in Qdrant with tags (`model: resnet`, `model: prithvi`), and let the UI query the database. This makes the UI switch instant and allows real-time benchmarking comparisons.

#### C. LLM Agent: Local (Ollama) vs. Free APIs (Gemini/Groq)
Since you have an RTX 4070 (12GB VRAM), you can run an 8-billion parameter model locally.
*   **Ollama (Local Llama-3-8B-Instruct or Mistral-7B):**
    *   *Pros:* Zero cost, complete privacy, huge "flex" for a CS project, immune to internet outages during presentation. Fits easily into 6-8GB VRAM using 4-bit/8-bit quantization.
    *   *Cons:* Context window is smaller (usually 8k), and agentic tool-calling (Text-to-SQL) can sometimes hallucinate more than GPT-4/Gemini.
*   **Free APIs (Google Gemini 1.5 Flash via AI Studio / Llama-3 via Groq):**
    *   *Pros:* Gemini gives you 1M context window for free (great for dumping massive NDAP CSVs). Groq is incredibly fast (800+ tokens/sec). Excellent at tool-calling.
    *   *Cons:* Rate limits (Groq is very strict on free tiers).
*   **Decision:** **Hybrid Design via LiteLLM**. Build a dropdown in the Streamlit UI: `[Engine: Local Llama-3 (Privacy Mode) | Cloud Gemini (High Context)]`.
    *   Use LangChain to define the agent. LangChain allows you to swap the underlying LLM object seamlessly. For the presentation, show that the system works fully air-gapped on your workstation, but can scale to Cloud APIs if needed.

---

### 3. The Multi-Modal Predictive Pipeline

To get an A+, you must prove your model is actually learning, not just guessing.

1.  **The Inputs:**
    *   *Visual:* Satellite embedding vectors (e.g., 512 dimensions from Prithvi).
    *   *Tabular:* Lagging NDAP features (e.g., 2019-2021 MGNREGA demand).
2.  **The Target (Label):**
    *   Current or recent NDAP target metric (e.g., 2024 NFHS standard of living index).
3.  **The Algorithm:**
    *   Concatenate the embeddings with the tabular data.
    *   Train an **XGBoost Regressor**.
    *   *Why XGBoost over a Neural Net?* XGBoost works brilliantly on tabular data + dense embeddings. More importantly, it allows you to use **SHAP (SHapley Additive exPlanations)**.
4.  **Explainability (Crucial for Grade):**
    *   In your UI, when a user clicks a district, show a SHAP waterfall plot. Explain *why* the model predicted a high poverty score (e.g., "Satellite Embedding Feature #42[Nightlight intensity] negatively impacted the score by -0.15").

---

### 4. Step-by-Step UI Flow (What the user/professor sees)

1.  **Sidebar:** Toggles for "Vision Model Selection" (ResNet vs Prithvi), "LLM Agent" (Ollama vs Gemini), and "Target Metric" (Electrification vs Wealth Index).
2.  **Main View - Top:** A large interactive 3D map of India (`pydeck`). Districts are color-coded based on the *Nowcasted* (predicted) values vs. *Historical* values. 
3.  **Main View - Bottom:** A Chat Interface.
    *   *User:* "Why is the predicted development score for district X so low, and what should the government do?"
    *   *Agent:* Queries PostGIS for historical data -> Queries XGBoost for the prediction -> Queries Qdrant for RAG on NITI Aayog policy docs -> *Generates response combining all three.*