import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from libpysal.weights import Queen
from esda.moran import Moran
import textwrap
import json
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(module_name=__name__, log_sub_dir="eda")

class EDAArtifactEngine:
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.eda_out = self.processed_dir / "eda_artifacts"
        self.eda_out.mkdir(parents=True, exist_ok=True)
        
        # We will focus the heavy spatial/temporal plots on these highly relevant targets
        # to avoid generating 100+ heavy spatial maps. 
        self.key_focus_metrics =[
            "hh_electricity",
            "hh_clean_cooking_fuel",
            "hh_improved_sanitation",
            "women_10yrs_schooling",
            "child_stunted",
            "women_anaemic",
            "institutional_births_total"
        ]

    @staticmethod
    def _wrap_text(text: str, width: int = 40) -> str:
        """Wraps long variable names with HTML breaks for Plotly readability."""
        if not isinstance(text, str):
            return text
        return "<br>".join(textwrap.wrap(text, width=width))

    def load_data(self):
        """Loads the processed tabular and spatial data."""
        logger.info("Loading processed data for EDA...")
        self.nfhs_long = pd.read_parquet(self.processed_dir / "tabular/nfhs_pooled_long.parquet")
        self.nfhs_wide = pd.read_parquet(self.processed_dir / "tabular/nfhs_pooled_wide.parquet")
        self.pmgsy = pd.read_parquet(self.processed_dir / "tabular/pmgsy_district_agg.parquet")
        self.mgnrega = pd.read_parquet(self.processed_dir / "tabular/mgnrega_state_clean.parquet")
        self.gdf = gpd.read_file(self.processed_dir / "spatial/india_districts_lgd.geojson")
        
        # Merge NFHS with spatial data for mapping
        self.gdf_2019 = self.gdf.merge(
            self.nfhs_wide[self.nfhs_wide['year'] == 2019], 
            on='district_lgd_code', 
            how='inner'
        )

    def generate_correlation_matrix(self):
        """Generates an interactive correlation heatmap for key indicators."""
        logger.info("Generating Correlation Matrix...")
        
        # Filter for focus metrics to keep the matrix readable
        focus_cols =[c for c in self.nfhs_wide.columns if any(k in c for k in self.key_focus_metrics)]
        if not focus_cols:
            focus_cols = self.nfhs_wide.select_dtypes(include='number').columns[:15] # Fallback
            
        corr_df = self.nfhs_wide[focus_cols].corr()
        
        # Clean labels for Plotly
        clean_labels = [self._wrap_text(c, 30) for c in corr_df.columns]
        
        fig = px.imshow(
            corr_df,
            x=clean_labels,
            y=clean_labels,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title="Correlation Heatmap of Key NFHS Indicators (Pooled Years)"
        )
        fig.update_layout(margin=dict(l=200, b=200)) # Extra margin for long names
        
        # Save Artifact
        fig.write_json(self.eda_out / "correlation_heatmap.json")

    def generate_temporal_drift(self):
        """Calculates delta between 2015 and 2019 and generates interactive drift plots."""
        logger.info("Generating Temporal Drift Visualizations...")
        
        df_2015 = self.nfhs_wide[self.nfhs_wide['year'] == 2015].set_index('district_lgd_code')
        df_2019 = self.nfhs_wide[self.nfhs_wide['year'] == 2019].set_index('district_lgd_code')
        
        # Calculate Delta (2019 - 2015)
        numeric_cols = df_2015.select_dtypes(include='number').columns.drop('year', errors='ignore')
        delta_df = df_2019[numeric_cols] - df_2015[numeric_cols]
        delta_df = delta_df.reset_index().merge(
            self.gdf[['district_lgd_code', 'district_name', 'state_name']], 
            on='district_lgd_code', how='inner'
        )
        
        # Save delta data for modeling
        delta_df.to_parquet(self.eda_out / "nfhs_delta_2015_2019.parquet", index=False)
        
        # Generate Box Plots for drift distribution
        for metric in self.key_focus_metrics:
            matched_cols =[c for c in numeric_cols if metric in c]
            if not matched_cols: continue
            target_col = matched_cols[0]
            
            fig = px.box(
                self.nfhs_long[self.nfhs_long['metric_name'] == target_col],
                x="year", y="metric_value", color="year",
                title=f"Distribution Shift (2015 vs 2019):<br>{self._wrap_text(target_col, 60)}",
                points="all", hover_data=["district", "state"]
            )
            fig.write_json(self.eda_out / f"drift_box_{target_col[:30]}.json")

    def generate_spatial_maps_and_stats(self):
        """Calculates Moran's I and generates Plotly Choropleths."""
        logger.info("Calculating Spatial Autocorrelation & Rendering Maps...")
        
        # Convert GeoJSON to format Plotly expects
        geojson_obj = json.loads(self.gdf_2019.to_json())
        
        # Calculate Spatial Weights (Queen contiguity)
        try:
            w = Queen.from_dataframe(self.gdf_2019)
            w.transform = 'r' # Row standardize
        except Exception as e:
            logger.warning(f"Could not compute spatial weights (island districts?): {e}")
            w = None

        spatial_stats = []

        for metric in self.key_focus_metrics:
            matched_cols =[c for c in self.gdf_2019.columns if metric in c]
            if not matched_cols: continue
            target_col = matched_cols[0]

            # 1. Plotly Interactive Choropleth Map
            fig = px.choropleth(
                self.gdf_2019,
                geojson=geojson_obj,
                locations='district_lgd_code',
                featureidkey="properties.district_lgd_code",
                color=target_col,
                hover_name='district_name',
                hover_data=['state_name', target_col],
                color_continuous_scale="Viridis",
                title=f"2019 Spatial Distribution:<br>{self._wrap_text(target_col, 60)}"
            )
            fig.update_geos(fitbounds="locations", visible=False)
            fig.write_json(self.eda_out / f"map_{target_col[:30]}.json")

            # 2. Moran's I Calculation (Is poverty/development clustered?)
            if w is not None:
                y = self.gdf_2019[target_col].fillna(self.gdf_2019[target_col].mean()).values
                moran = Moran(y, w)
                spatial_stats.append({
                    "metric": target_col,
                    "morans_i": moran.I,
                    "p_value": moran.p_sim
                })

        if spatial_stats:
            pd.DataFrame(spatial_stats).to_csv(self.eda_out / "morans_i_spatial_stats.csv", index=False)
            logger.info("Saved Moran's I spatial statistics.")

    def generate_macro_micro_scatter(self):
        """Scatters Infrastructure (PMGSY/MGNREGA) against NFHS metrics."""
        logger.info("Generating Macro-Micro Feature Scatters...")
        
        # Merge PMGSY (District) and MGNREGA (State) into NFHS 2019
        df_2019 = self.nfhs_wide[self.nfhs_wide['year'] == 2019]
        merged = df_2019.merge(self.pmgsy[self.pmgsy['year'] == 2019], on='district_lgd_code', how='left')
        merged = merged.merge(self.mgnrega[self.mgnrega['year'] == 2019], on='state_lgd_code', how='left')
        
        # Ensure we have target column
        electricity_col =[c for c in df_2019.columns if "electricity" in c]
        if electricity_col:
            fig = px.scatter(
                merged, 
                x="pmgsy_road_length_km", 
                y=electricity_col[0],
                color="state_x", # state name
                hover_name="district",
                trendline="ols",
                title="Infrastructure vs Development: Road Length vs Electrification (2019)",
                labels={
                    "pmgsy_road_length_km": "PMGSY Roads Built (km)",
                    electricity_col[0]: self._wrap_text(electricity_col[0], 40)
                }
            )
            fig.write_json(self.eda_out / "scatter_pmgsy_vs_electricity.json")

    def run_pipeline(self):
        logger.info("--- Starting EDA Artifact Pipeline ---")
        self.load_data()
        self.generate_correlation_matrix()
        self.generate_temporal_drift()
        self.generate_spatial_maps_and_stats()
        self.generate_macro_micro_scatter()
        logger.info(f"--- EDA Pipeline Complete. Artifacts saved to {self.eda_out} ---")

if __name__ == "__main__":
    engine = EDAArtifactEngine()
    engine.run_pipeline()