import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import json
import plotly.io as pio
import plotly.express as px
import sys
from pathlib import Path

# Resolve paths relative to the repository root, regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import our Agent
from src.agent.llm_agent import IndiSightAgent
from src.utils.logger import get_logger
from src.utils.contract_validator import BenchmarkContract

logger = get_logger(module_name=__name__, log_sub_dir="app")

# --- Configuration & Styling ---
st.set_page_config(page_title="IndiSight Nowcast", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #FAFAFA;}
    .stChatInputContainer {padding-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# --- Caching Data & Agent ---
@st.cache_data
def load_data():
    """Loads and caches tabular and spatial data."""
    logger.info("Loading dashboard data artifacts from data/processed...")
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    # Load Long Tabular Data
    nfhs_long = pd.read_parquet(processed_dir / "tabular/nfhs_pooled_long.parquet")
    
    # Load GeoJSON
    gdf = gpd.read_file(processed_dir / "spatial/india_districts_lgd.geojson")
    
    # Pre-compute available metrics
    available_metrics = sorted(nfhs_long['metric_name'].unique().tolist())
    logger.info(f"Loaded dashboard data: nfhs_long={len(nfhs_long)} rows, metrics={len(available_metrics)}")

    return nfhs_long, gdf, available_metrics

@st.cache_resource
def get_agent():
    """Initializes the LangChain Agent once."""
    logger.info("Initializing cached IndiSight agent for Streamlit session")
    return IndiSightAgent(llm_mode="gemini")

@st.cache_data
def get_benchmarks():
    validator = BenchmarkContract(PROJECT_ROOT / "data" / "processed" / "benchmarks")
    return validator.get_valid_benchmarks()

@st.cache_data
def load_predictions(predictions_path):
    return pd.read_parquet(predictions_path)

@st.cache_data
def load_shap(shap_path):
    with open(shap_path, "r") as f:
        return json.load(f)

@st.cache_data
def load_plotly_artifact(filename: str):
    """Loads a pre-rendered Plotly JSON artifact."""
    filepath = PROJECT_ROOT / "data" / "processed" / "eda_artifacts" / filename
    if filepath.exists():
        with open(filepath, "r") as f:
            return pio.from_json(f.read())
    return None

def render_plotly(fig, key: str, use_container_width: bool = True):
    """Render Plotly charts with stable Streamlit keys."""
    st.plotly_chart(fig, use_container_width=use_container_width, key=key)

@st.cache_data
def list_eda_artifacts():
    """List available EDA artifact files for dynamic dashboards."""
    artifact_dir = PROJECT_ROOT / "data" / "processed" / "eda_artifacts"
    map_files = sorted([p.name for p in artifact_dir.glob("map_*.json")])
    drift_files = sorted([p.name for p in artifact_dir.glob("drift_box_*.json")])
    distribution_files = sorted([p.name for p in artifact_dir.glob("distribution_*.json")])
    trend_files = sorted([p.name for p in artifact_dir.glob("trend_*.json")])
    ranking_files = sorted([p.name for p in artifact_dir.glob("ranking_*.json")])
    scatter_files = sorted([p.name for p in artifact_dir.glob("scatter_*.json")])
    artifacts = {
        "map_files": map_files,
        "drift_files": drift_files,
        "distribution_files": distribution_files,
        "trend_files": trend_files,
        "ranking_files": ranking_files,
        "scatter_files": scatter_files,
        "morans_csv": (artifact_dir / "morans_i_spatial_stats.csv"),
        "dataset_summary_csv": (artifact_dir / "eda_dataset_summary.csv"),
        "metric_quality_csv": (artifact_dir / "metric_quality_by_year.csv"),
        "missingness_csv": (artifact_dir / "metric_missingness.csv"),
        "outlier_csv": (artifact_dir / "outlier_summary_2019.csv"),
        "state_rankings_csv": (artifact_dir / "state_metric_rankings_2019.csv"),
    }
    logger.info(
        "EDA artifact scan complete: maps=%s drift=%s trend=%s distribution=%s ranking=%s scatter=%s",
        len(map_files),
        len(drift_files),
        len(trend_files),
        len(distribution_files),
        len(ranking_files),
        len(scatter_files),
    )
    return artifacts

def validate_eda_contract(artifacts: dict) -> list[str]:
    """Validate required EDA artifacts expected by the dashboard contract."""
    missing = []

    required_csv = {
        "morans_csv": "morans_i_spatial_stats.csv",
        "dataset_summary_csv": "eda_dataset_summary.csv",
        "metric_quality_csv": "metric_quality_by_year.csv",
        "missingness_csv": "metric_missingness.csv",
        "outlier_csv": "outlier_summary_2019.csv",
        "state_rankings_csv": "state_metric_rankings_2019.csv",
    }

    for key, name in required_csv.items():
        path_obj = artifacts.get(key)
        if not path_obj or not path_obj.exists():
            missing.append(name)

    if not load_plotly_artifact("correlation_heatmap.json"):
        missing.append("correlation_heatmap.json")

    if not artifacts.get("map_files"):
        missing.append("map_*.json")
    if not artifacts.get("drift_files"):
        missing.append("drift_box_*.json")
    if not artifacts.get("trend_files"):
        missing.append("trend_*.json")
    if not artifacts.get("distribution_files"):
        missing.append("distribution_*.json")
    if not artifacts.get("ranking_files"):
        missing.append("ranking_*.json")
    if not artifacts.get("scatter_files"):
        missing.append("scatter_*.json")

    return missing

# --- Initialization ---
nfhs_long, gdf, available_metrics = load_data()
valid_benchmarks = get_benchmarks()
agent = get_agent()

# --- Sidebar UI ---
st.sidebar.title("IndiSight Controls")

st.sidebar.header("1. Map Configuration")
selected_metric = st.sidebar.selectbox("Target Metric", available_metrics, index=available_metrics.index("hh_electricity") if "hh_electricity" in available_metrics else 0)
selected_year = st.sidebar.radio("Data Year", [2015, 2019], index=1)

st.sidebar.header("2. Model Configuration")
selected_run_data = None
predictions_df = None
shap_data = None

if not valid_benchmarks:
    st.sidebar.warning("No valid benchmarks found. Pending Member A's artifacts.")
    st.sidebar.selectbox("Vision Model",["ResNet-50", "Prithvi-100M (EO)"], disabled=True, help="Will be enabled when Member A completes embeddings.")
    st.sidebar.slider("PCA Dimensions", min_value=16, max_value=1024, value=128, step=16, disabled=True)
else:
    run_ids = [b["run_id"] for b in valid_benchmarks]
    selected_run_id = st.sidebar.selectbox("Benchmark Run", run_ids)
    selected_run_data = next((b for b in valid_benchmarks if b["run_id"] == selected_run_id), None)
    if selected_run_data:
        predictions_df = load_predictions(selected_run_data["paths"]["predictions"])
        shap_data = load_shap(selected_run_data["paths"]["shap"])

# --- Main UI Layout ---
st.title("IndiSight: Multi-Modal Economic Nowcasting")
st.markdown("Fusing Satellite Imagery, PMGSY Infrastructure, and NDAP Tabular Data to estimate district-level development in real-time.")

tab_map, tab_chat, tab_eda = st.tabs(["3D Spatial View", "AI Policy Assistant", "Metrics & EDA"])

# -----------------------------------------
# TAB 1: 3D Spatial Map (PyDeck)
# -----------------------------------------
with tab_map:
    st.subheader(f"Spatial Distribution: {selected_metric} ({selected_year})")
    
    map_mode = st.radio("View Mode", ["Historical Data", "Prediction Delta"] if predictions_df is not None else ["Historical Data"], horizontal=True)
    
    # Filter Data
    df_filtered = nfhs_long[(nfhs_long['metric_name'] == selected_metric) & (nfhs_long['year'] == selected_year)]
    
    if df_filtered.empty:
        st.warning(f"No data available for {selected_metric} in {selected_year}.")
    else:
        # Merge with GeoJSON
        gdf_merged = gdf.merge(df_filtered[['district_lgd_code', 'metric_value']], on='district_lgd_code', how='left')
        gdf_merged['metric_value'] = gdf_merged['metric_value'].fillna(0)
        
        display_col = 'metric_value'
        if map_mode == "Prediction Delta" and predictions_df is not None:
            gdf_merged = gdf_merged.merge(predictions_df[['district_lgd_code', 'delta']], on='district_lgd_code', how='left')
            gdf_merged['delta'] = gdf_merged['delta'].fillna(0)
            display_col = 'delta'
            
        # Calculate max value for extrusion scaling
        max_val = gdf_merged[display_col].abs().max()
        scale_factor = 100000 / (max_val if max_val > 0 else 1)
        
        # Pre-compute elevation and color in pandas to avoid JS evaluation errors in PyDeck
        gdf_merged['elevation'] = gdf_merged[display_col].abs() * scale_factor
        
        def get_color(val):
            if map_mode == "Prediction Delta":
                return [0, 255, 0, 150] if val > 0 else [255, 0, 0, 150]
            else:
                ratio = val / (max_val if max_val > 0 else 1)
                return [int(255 - (ratio * 255)), int(ratio * 200), 150, 150]
                
        gdf_merged['fill_color'] = gdf_merged[display_col].apply(get_color)
        
        # Convert to JSON for PyDeck
        geojson_data = json.loads(gdf_merged.to_json())
        
        # PyDeck Configuration
        layer = pdk.Layer(
            "GeoJsonLayer",
            geojson_data,
            opacity=0.8,
            stroked=False,
            filled=True,
            extruded=True,
            wireframe=True,
            get_elevation="properties.elevation",
            get_fill_color="properties.fill_color",
            get_line_color=[255, 255, 255],
            pickable=True,
        )

        # Initial View State over Central India
        view_state = pdk.ViewState(
            latitude=21.1458,
            longitude=79.0882,
            zoom=4,
            pitch=45,
            bearing=0
        )

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "{district_name} ({state_name})\nValue: {metric_value}\nDelta: {delta}" if map_mode == "Prediction Delta" else "{district_name} ({state_name})\nValue: {metric_value}"},
            map_style=pdk.map_styles.DARK
        )

        st.pydeck_chart(r)
        
    if predictions_df is not None and shap_data is not None:
        st.markdown("---")
        st.subheader("District-Level Explainability (SHAP)")
        
        districts = predictions_df.merge(gdf[['district_lgd_code', 'district_name']], on='district_lgd_code', how='left')
        district_options = dict(zip(districts['district_lgd_code'], districts['district_name']))
        selected_district_code = st.selectbox("Select District for Deep Dive", options=list(district_options.keys()), format_func=lambda x: district_options[x])
        
        if selected_district_code:
            str_code = str(selected_district_code)
            if str_code in shap_data:
                features = shap_data[str_code]
                if features:
                    import plotly.graph_objects as go
                    feature_names = [f["feature"] for f in features]
                    shap_values = [f["shap_value"] for f in features]
                    
                    fig = go.Figure(go.Waterfall(
                        orientation="h",
                        measure=["relative"] * len(features),
                        y=feature_names,
                        x=shap_values,
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    ))
                    fig.update_layout(title="SHAP Waterfall Plot", waterfallgap=0.3)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No SHAP features available for this district.")
            else:
                st.info("SHAP data not available for this district.")

# -----------------------------------------
# TAB 2: AI Policy Assistant
# -----------------------------------------
with tab_chat:
    st.subheader("Query Database & Policy Documents")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages =[]
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("thinking"):
                with st.expander("Show reasoning"):
                    st.markdown(message["thinking"])

    # Process queued prompt before rendering the input widget so the input stays at the bottom.
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        logger.info("Processing queued assistant prompt")
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ""
                thinking = ""
                answer_placeholder = st.empty()
                reasoning_container = st.empty()
                thinking_placeholder = None

                for update in agent.stream_chat_with_details(prompt):
                    if update.get("thinking"):
                        thinking = update["thinking"]
                        if thinking_placeholder is None:
                            with reasoning_container.container():
                                with st.expander("Show reasoning", expanded=True):
                                    thinking_placeholder = st.empty()
                        thinking_placeholder.markdown(thinking)

                    if update.get("done"):
                        response = update.get("answer", "")
                        answer_placeholder.markdown(response)

                if not response:
                    response = "I didn't receive a response from the agent."
                    answer_placeholder.markdown(response)
                logger.info("Assistant response generated")

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "thinking": thinking}
        )
        st.session_state.pending_prompt = None
        st.rerun()

    # React to user input
    if prompt := st.chat_input("Ask about NDAP data or PMGSY guidelines..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_prompt = prompt
        logger.info("Queued user prompt for assistant processing")
        st.rerun()

# -----------------------------------------
# TAB 3: EDA & Benchmarks
# -----------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    artifacts = list_eda_artifacts()
    contract_missing = validate_eda_contract(artifacts)

    if contract_missing:
        logger.warning(f"EDA contract missing artifacts: {contract_missing}")
        st.warning(
            "EDA artifact contract is incomplete. Missing: "
            + ", ".join(contract_missing)
            + ". Run the EDA pipeline to regenerate full coverage."
        )

    st.caption(
        (
            f"Artifact coverage: {len(artifacts['map_files'])} maps, "
            f"{len(artifacts['drift_files'])} drift plots, "
            f"{len(artifacts['trend_files'])} trend plots, "
            f"{len(artifacts['distribution_files'])} distribution plots, "
            f"{len(artifacts['ranking_files'])} ranking plots, "
            f"{len(artifacts['scatter_files'])} macro-micro scatter plots"
        )
    )

    st.markdown("### Data Quality and Coverage")
    dq_col1, dq_col2 = st.columns(2)

    with dq_col1:
        if artifacts["dataset_summary_csv"].exists():
            st.markdown("Dataset Summary")
            st.dataframe(pd.read_csv(artifacts["dataset_summary_csv"]), use_container_width=True)
        else:
            st.info("Dataset summary not found. Re-run EDA pipeline.")

    with dq_col2:
        if artifacts["missingness_csv"].exists():
            missing_df = pd.read_csv(artifacts["missingness_csv"])
            st.markdown("Top Missing Metrics")
            st.dataframe(missing_df.head(15), use_container_width=True)
        else:
            st.info("Missingness report not found. Re-run EDA pipeline.")

    if artifacts["outlier_csv"].exists():
        st.markdown("Outlier Summary (2019)")
        st.dataframe(pd.read_csv(artifacts["outlier_csv"]), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        corr_fig = load_plotly_artifact("correlation_heatmap.json")
        if corr_fig:
            render_plotly(corr_fig, key="eda_corr_heatmap")
        else:
            st.info("Correlation heatmap not found. Run EDA pipeline.")
            
    with col2:
        scatter_fig = load_plotly_artifact("scatter_pmgsy_vs_electricity.json")
        if scatter_fig:
            render_plotly(scatter_fig, key="eda_scatter_canonical")
        else:
            st.info("Road length vs electrification plot not found or empty. Run EDA pipeline.")

    st.markdown("### Macro-Micro Scatter Explorer")
    if artifacts["scatter_files"]:
        default_scatter = (
            "scatter_pmgsy_vs_electricity.json"
            if "scatter_pmgsy_vs_electricity.json" in artifacts["scatter_files"]
            else artifacts["scatter_files"][0]
        )
        selected_scatter = st.selectbox(
            "Choose infrastructure-vs-development scatter",
            artifacts["scatter_files"],
            index=artifacts["scatter_files"].index(default_scatter),
            key="eda_scatter_selector",
        )
        scatter_sel_fig = load_plotly_artifact(selected_scatter)
        if scatter_sel_fig:
            render_plotly(scatter_sel_fig, key=f"eda_scatter_{selected_scatter}")
    else:
        st.info("No scatter artifacts found in data/processed/eda_artifacts.")

    st.markdown("### National Trend Explorer")
    if artifacts["trend_files"]:
        selected_trend = st.selectbox(
            "Choose national trend artifact",
            artifacts["trend_files"],
            key="eda_trend_selector",
        )
        trend_fig = load_plotly_artifact(selected_trend)
        if trend_fig:
            render_plotly(trend_fig, key=f"eda_trend_{selected_trend}")
    else:
        st.info("No trend artifacts found in data/processed/eda_artifacts.")

    st.markdown("### Distribution Profile Explorer")
    if artifacts["distribution_files"]:
        selected_distribution = st.selectbox(
            "Choose 2019 distribution artifact",
            artifacts["distribution_files"],
            key="eda_distribution_selector",
        )
        dist_fig = load_plotly_artifact(selected_distribution)
        if dist_fig:
            render_plotly(dist_fig, key=f"eda_distribution_{selected_distribution}")
    else:
        st.info("No distribution artifacts found in data/processed/eda_artifacts.")
            
    st.markdown("### Spatial Map Explorer")
    if artifacts["map_files"]:
        default_map = (
            "map_hh_electricity.json"
            if "map_hh_electricity.json" in artifacts["map_files"]
            else artifacts["map_files"][0]
        )
        selected_map = st.selectbox(
            "Choose a district-level map artifact",
            artifacts["map_files"],
            index=artifacts["map_files"].index(default_map),
            key="eda_map_selector",
        )
        map_fig = load_plotly_artifact(selected_map)
        if map_fig:
            render_plotly(map_fig, key=f"eda_map_{selected_map}")
    else:
        st.info("No map artifacts found in data/processed/eda_artifacts.")

    st.markdown("### Temporal Drift Explorer (2015 vs 2019)")
    if artifacts["drift_files"]:
        default_drift = (
            "drift_box_hh_electricity.json"
            if "drift_box_hh_electricity.json" in artifacts["drift_files"]
            else artifacts["drift_files"][0]
        )
        selected_drift = st.selectbox(
            "Choose a drift artifact",
            artifacts["drift_files"],
            index=artifacts["drift_files"].index(default_drift),
            key="eda_drift_selector",
        )
        drift_fig = load_plotly_artifact(selected_drift)
        if drift_fig:
            render_plotly(drift_fig, key=f"eda_drift_{selected_drift}")
    else:
        st.info("No drift artifacts found in data/processed/eda_artifacts.")

    st.markdown("### Spatial Autocorrelation (Moran's I)")
    morans_path = artifacts["morans_csv"]
    if morans_path.exists():
        morans_df = pd.read_csv(morans_path)
        st.dataframe(morans_df, use_container_width=True)
        if {"metric_name", "morans_i"}.issubset(set(morans_df.columns)):
            bar_fig = px.bar(
                morans_df,
                x="metric_name",
                y="morans_i",
                color="morans_i",
                title="Moran's I by Metric",
            )
            bar_fig.update_layout(xaxis_title="Metric", yaxis_title="Moran's I")
            render_plotly(bar_fig, key="eda_morans_i")
    else:
        st.info("Moran's I CSV not found in data/processed/eda_artifacts.")

    st.markdown("### State Ranking Explorer (2019)")
    if artifacts["ranking_files"]:
        selected_ranking = st.selectbox(
            "Choose top/bottom state ranking artifact",
            artifacts["ranking_files"],
            key="eda_ranking_selector",
        )
        ranking_fig = load_plotly_artifact(selected_ranking)
        if ranking_fig:
            render_plotly(ranking_fig, key=f"eda_ranking_{selected_ranking}")
    else:
        st.info("No ranking artifacts found in data/processed/eda_artifacts.")

    if artifacts["state_rankings_csv"].exists():
        with st.expander("View Full State Ranking Table"):
            st.dataframe(pd.read_csv(artifacts["state_rankings_csv"]), use_container_width=True)

    if artifacts["metric_quality_csv"].exists():
        with st.expander("View Metric Quality by Year"):
            st.dataframe(pd.read_csv(artifacts["metric_quality_csv"]), use_container_width=True)