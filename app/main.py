import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import json
import plotly.io as pio
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
            tooltip={"text": "{district_name} ({state_name})\nValue: {metric_value}"},
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

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask about NDAP data or PMGSY guidelines..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.chat(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# -----------------------------------------
# TAB 3: EDA & Benchmarks
# -----------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        corr_fig = load_plotly_artifact("correlation_heatmap.json")
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.info("Correlation heatmap not found. Run EDA pipeline.")
            
    with col2:
        scatter_fig = load_plotly_artifact("scatter_pmgsy_vs_electricity.json")
        if scatter_fig:
            st.plotly_chart(scatter_fig, use_container_width=True)
        else:
            st.info("Scatter plot not found. Run EDA pipeline.")
            
    st.markdown("### Temporal Drift (2015 vs 2019)")
    drift_fig = load_plotly_artifact("drift_box_hh_electricity.json")
    if drift_fig:
        st.plotly_chart(drift_fig, use_container_width=True)