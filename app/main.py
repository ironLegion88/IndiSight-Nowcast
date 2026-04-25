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
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    # Load Long Tabular Data
    nfhs_long = pd.read_parquet(processed_dir / "tabular/nfhs_pooled_long.parquet")
    
    # Load GeoJSON
    gdf = gpd.read_file(processed_dir / "spatial/india_districts_lgd.geojson")
    
    # Pre-compute available metrics
    available_metrics = sorted(nfhs_long['metric_name'].unique().tolist())
    
    return nfhs_long, gdf, available_metrics

@st.cache_resource
def get_agent():
    """Initializes the LangChain Agent once."""
    return IndiSightAgent(llm_mode="gemini")

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
    return {
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

# --- Initialization ---
nfhs_long, gdf, available_metrics = load_data()
agent = get_agent()

# --- Sidebar UI ---
st.sidebar.title("IndiSight Controls")

st.sidebar.header("1. Map Configuration")
selected_metric = st.sidebar.selectbox("Target Metric", available_metrics, index=available_metrics.index("hh_electricity") if "hh_electricity" in available_metrics else 0)
selected_year = st.sidebar.radio("Data Year", [2015, 2019], index=1)

st.sidebar.header("2. Model Configuration (Pending Member A)")
st.sidebar.selectbox("Vision Model",["ResNet-50", "Prithvi-100M (EO)"], disabled=True, help="Will be enabled when Member A completes embeddings.")
st.sidebar.slider("PCA Dimensions", min_value=16, max_value=1024, value=128, step=16, disabled=True)

# --- Main UI Layout ---
st.title("IndiSight: Multi-Modal Economic Nowcasting")
st.markdown("Fusing Satellite Imagery, PMGSY Infrastructure, and NDAP Tabular Data to estimate district-level development in real-time.")

tab_map, tab_chat, tab_eda = st.tabs(["3D Spatial View", "AI Policy Assistant", "Metrics & EDA"])

# -----------------------------------------
# TAB 1: 3D Spatial Map (PyDeck)
# -----------------------------------------
with tab_map:
    st.subheader(f"Spatial Distribution: {selected_metric} ({selected_year})")
    
    # Filter Data
    df_filtered = nfhs_long[(nfhs_long['metric_name'] == selected_metric) & (nfhs_long['year'] == selected_year)]
    
    if df_filtered.empty:
        st.warning(f"No data available for {selected_metric} in {selected_year}.")
    else:
        # Merge with GeoJSON
        gdf_merged = gdf.merge(df_filtered[['district_lgd_code', 'metric_value']], on='district_lgd_code', how='left')
        gdf_merged['metric_value'] = gdf_merged['metric_value'].fillna(0)
        
        # Calculate max value for extrusion scaling
        max_val = gdf_merged['metric_value'].max()
        scale_factor = 100000 / (max_val if max_val > 0 else 1)
        
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
            get_elevation=f"properties.metric_value * {scale_factor}",
            get_fill_color="[255 - (properties.metric_value / {max_val} * 255), properties.metric_value / {max_val} * 200, 150]",
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
            tooltip={"text": "{district_name} ({state_name})\nValue: {metric_value}"},  # pyright: ignore[reportArgumentType]
            map_style=pdk.map_styles.DARK
        )

        st.pydeck_chart(r)

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

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "thinking": thinking}
        )
        st.session_state.pending_prompt = None
        st.rerun()

    # React to user input
    if prompt := st.chat_input("Ask about NDAP data or PMGSY guidelines..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_prompt = prompt
        st.rerun()

# -----------------------------------------
# TAB 3: EDA & Benchmarks
# -----------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    artifacts = list_eda_artifacts()

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