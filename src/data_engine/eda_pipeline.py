import pandas as pd
import geopandas as gpd
import plotly.express as px
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
        self.profile_top_n = 12
        
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

    def _ensure_focus_metrics(self):
        """Resolve configured focus metrics against available NFHS long metric names."""
        available = set(self.nfhs_long["metric_name"].unique())
        resolved = [m for m in self.key_focus_metrics if m in available]
        if not resolved:
            resolved = (
                self.nfhs_long["metric_name"]
                .value_counts()
                .head(self.profile_top_n)
                .index
                .tolist()
            )
        return resolved

    def _safe_filename(self, name: str) -> str:
        return str(name).replace("/", "_").replace(" ", "_")[:60]

    @staticmethod
    def _require_columns(df: pd.DataFrame, required: list[str], dataset_name: str) -> None:
        """Raise a clear error if an ingestion dump does not match expected structure."""
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"{dataset_name} is missing required columns: {missing}. "
                "Re-run src.data_engine.ingest_tabular to regenerate compatible dumps."
            )

    def load_data(self):
        """Loads the processed tabular and spatial data."""
        logger.info("Loading processed data for EDA...")
        self.nfhs_long = pd.read_parquet(self.processed_dir / "tabular/nfhs_pooled_long.parquet")
        self.nfhs_wide = pd.read_parquet(self.processed_dir / "tabular/nfhs_pooled_wide.parquet")
        self.pmgsy = pd.read_parquet(self.processed_dir / "tabular/pmgsy_district_agg.parquet")
        self.mgnrega = pd.read_parquet(self.processed_dir / "tabular/mgnrega_state_clean.parquet")
        self.gdf = gpd.read_file(self.processed_dir / "spatial/india_districts_lgd.geojson")

        # Validate compatibility with the ingestion dump schema.
        self._require_columns(
            self.nfhs_long,
            ["state_lgd_code", "state", "district_lgd_code", "district", "year", "metric_name", "metric_value"],
            "nfhs_pooled_long.parquet",
        )
        self._require_columns(
            self.nfhs_wide,
            ["state_lgd_code", "state", "district_lgd_code", "district", "year"],
            "nfhs_pooled_wide.parquet",
        )
        self._require_columns(
            self.pmgsy,
            ["district_lgd_code", "year", "pmgsy_road_length_km", "pmgsy_sanction_cost"],
            "pmgsy_district_agg.parquet",
        )
        self._require_columns(
            self.mgnrega,
            ["state_lgd_code", "state", "year", "mgnrega_demand_households", "mgnrega_labour_exp"],
            "mgnrega_state_clean.parquet",
        )
        self._require_columns(
            self.gdf,
            ["district_lgd_code", "district_name", "state_name"],
            "india_districts_lgd.geojson",
        )
        
        # Merge NFHS with spatial data for mapping
        self.gdf_2019 = self.gdf.merge(
            self.nfhs_wide[self.nfhs_wide['year'] == 2019], 
            on='district_lgd_code', 
            how='inner'
        )

        self.focus_metrics_resolved = self._ensure_focus_metrics()

    def generate_data_quality_reports(self):
        """Generate dataset-level and metric-level quality diagnostics."""
        logger.info("Generating EDA quality diagnostics...")

        summary_rows = [
            {"dataset": "nfhs_long", "rows": len(self.nfhs_long), "columns": len(self.nfhs_long.columns)},
            {"dataset": "nfhs_wide", "rows": len(self.nfhs_wide), "columns": len(self.nfhs_wide.columns)},
            {"dataset": "pmgsy", "rows": len(self.pmgsy), "columns": len(self.pmgsy.columns)},
            {"dataset": "mgnrega", "rows": len(self.mgnrega), "columns": len(self.mgnrega.columns)},
        ]
        pd.DataFrame(summary_rows).to_csv(self.eda_out / "eda_dataset_summary.csv", index=False)

        metric_quality = (
            self.nfhs_long.groupby(["metric_name", "year"])
            .agg(
                n_records=("metric_value", "count"),
                mean=("metric_value", "mean"),
                std=("metric_value", "std"),
                min=("metric_value", "min"),
                max=("metric_value", "max"),
            )
            .reset_index()
        )
        metric_quality.to_csv(self.eda_out / "metric_quality_by_year.csv", index=False)

        id_cols = ["state_lgd_code", "state", "district_lgd_code", "district", "year"]
        metric_cols = [c for c in self.nfhs_wide.columns if c not in id_cols]
        missingness = self.nfhs_wide[metric_cols].isna().mean().sort_values(ascending=False)
        missingness_df = missingness.rename("missing_rate").reset_index().rename(columns={"index": "metric_name"})
        missingness_df.to_csv(self.eda_out / "metric_missingness.csv", index=False)

        fig = px.bar(
            missingness_df.head(30),
            x="metric_name",
            y="missing_rate",
            color="missing_rate",
            title="Top 30 Metrics by Missingness",
            labels={"missing_rate": "Missing Fraction"},
        )
        fig.update_layout(xaxis_tickangle=45)
        fig.write_json(self.eda_out / "missingness_top30.json")

    def generate_national_profiles(self):
        """Generate rigorous univariate and temporal profiles for focus metrics."""
        logger.info("Generating national metric profiles...")

        outlier_rows = []
        for metric in self.focus_metrics_resolved:
            metric_df = self.nfhs_long[self.nfhs_long["metric_name"] == metric].copy()
            metric_2019 = metric_df[metric_df["year"] == 2019].copy()
            if metric_2019.empty:
                continue

            q1 = metric_2019["metric_value"].quantile(0.25)
            q3 = metric_2019["metric_value"].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((metric_2019["metric_value"] < lower) | (metric_2019["metric_value"] > upper)).sum())
            outlier_rows.append(
                {
                    "metric_name": metric,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "outlier_count_2019": outlier_count,
                }
            )

            dist_fig = px.histogram(
                metric_2019,
                x="metric_value",
                nbins=40,
                marginal="box",
                title=f"2019 Distribution Profile: {self._wrap_text(metric, 50)}",
            )
            dist_fig.write_json(self.eda_out / f"distribution_{self._safe_filename(metric)}.json")

            trend = (
                metric_df.groupby("year")["metric_value"]
                .agg(mean="mean", median="median", std="std", min="min", max="max")
                .reset_index()
            )
            trend_fig = px.line(
                trend,
                x="year",
                y=["mean", "median"],
                markers=True,
                title=f"National Trend: {self._wrap_text(metric, 50)}",
            )
            trend_fig.write_json(self.eda_out / f"trend_{self._safe_filename(metric)}.json")

        if outlier_rows:
            pd.DataFrame(outlier_rows).to_csv(self.eda_out / "outlier_summary_2019.csv", index=False)

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
        for metric in self.focus_metrics_resolved:
            matched_cols =[c for c in numeric_cols if metric in c]
            if not matched_cols: continue
            target_col = matched_cols[0]
            
            fig = px.box(
                self.nfhs_long[self.nfhs_long['metric_name'] == target_col],
                x="year", y="metric_value", color="year",
                title=f"Distribution Shift (2015 vs 2019):<br>{self._wrap_text(target_col, 60)}",
                points="all", hover_data=["district", "state"]
            )
            fig.write_json(self.eda_out / f"drift_box_{self._safe_filename(target_col)}.json")

    def generate_spatial_maps_and_stats(self):
        """Calculates Moran's I and generates Plotly Choropleths."""
        logger.info("Calculating Spatial Autocorrelation & Rendering Maps...")
        
        # Convert GeoJSON to format Plotly expects
        geojson_obj = json.loads(self.gdf_2019.to_json())
        
        # Calculate Spatial Weights (Queen contiguity)
        try:
            w = Queen.from_dataframe(self.gdf_2019)
            w.transform = 'r' # Row standardize  # pyright: ignore[reportAttributeAccessIssue]
        except Exception as e:
            logger.warning(f"Could not compute spatial weights (island districts?): {e}")
            w = None

        spatial_stats = []

        for metric in self.focus_metrics_resolved:
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
            fig.write_json(self.eda_out / f"map_{self._safe_filename(target_col)}.json")

            # 2. Moran's I Calculation (Is poverty/development clustered?)
            if w is not None:
                y = self.gdf_2019[target_col].fillna(self.gdf_2019[target_col].mean()).values
                moran = Moran(y, w)
                spatial_stats.append({
                    "metric_name": target_col,
                    "morans_i": moran.I,
                    "p_value": moran.p_sim
                })

        if spatial_stats:
            pd.DataFrame(spatial_stats).to_csv(self.eda_out / "morans_i_spatial_stats.csv", index=False)
            logger.info("Saved Moran's I spatial statistics.")

    def generate_state_rankings(self):
        """Generate top/bottom state ranking charts for focus metrics in 2019."""
        logger.info("Generating state ranking views...")

        df_2019 = self.nfhs_long[self.nfhs_long["year"] == 2019].copy()
        if df_2019.empty:
            return

        ranking_records = []
        for metric in self.focus_metrics_resolved:
            mdf = df_2019[df_2019["metric_name"] == metric].copy()
            if mdf.empty:
                continue

            state_mean = (
                mdf.groupby("state", as_index=False)["metric_value"]
                .mean()
                .sort_values("metric_value", ascending=False)
            )
            state_mean["metric_name"] = metric
            ranking_records.append(state_mean)

            top_bottom = pd.concat([state_mean.head(8), state_mean.tail(8)]).drop_duplicates()
            fig = px.bar(
                top_bottom.sort_values("metric_value"),
                x="metric_value",
                y="state",
                orientation="h",
                color="metric_value",
                title=f"Top/Bottom States (2019): {self._wrap_text(metric, 50)}",
            )
            fig.write_json(self.eda_out / f"ranking_{self._safe_filename(metric)}.json")

        if ranking_records:
            pd.concat(ranking_records, ignore_index=True).to_csv(
                self.eda_out / "state_metric_rankings_2019.csv", index=False
            )

    def generate_macro_micro_scatter(self):
        """Scatters infrastructure/governance features against NFHS focus metrics."""
        logger.info("Generating Macro-Micro Feature Scatters...")
        
        # Merge cumulative PMGSY (District) up to 2019 and MGNREGA (State) for 2019 into NFHS 2019
        df_2019 = self.nfhs_wide[self.nfhs_wide['year'] == 2019]
        pmgsy_cumulative = self.pmgsy[self.pmgsy['year'] <= 2019].groupby('district_lgd_code')[["pmgsy_road_length_km", "pmgsy_sanction_cost"]].sum().reset_index()
        merged = df_2019.merge(pmgsy_cumulative, on='district_lgd_code', how='left')
        merged = merged.merge(self.mgnrega[self.mgnrega['year'] == 2019], on='state_lgd_code', how='left')
        
        # unify state column name after merges
        if "state_x" in merged.columns and "state" not in merged.columns:
            merged = merged.rename(columns={"state_x": "state"})

        macro_vars = [
            "pmgsy_road_length_km",
            "pmgsy_sanction_cost",
            "mgnrega_demand_households",
            "mgnrega_labour_exp",
        ]

        for metric in self.focus_metrics_resolved:
            y_candidates = [c for c in merged.columns if metric in c]
            if not y_candidates:
                continue
            target_col = y_candidates[0]

            for macro_var in macro_vars:
                if macro_var not in merged.columns:
                    continue

                plot_df = merged[[macro_var, target_col, "district", "state"]].dropna()
                if len(plot_df) < 20:
                    continue

                fig = px.scatter(
                    plot_df,
                    x=macro_var,
                    y=target_col,
                    color="state",
                    hover_name="district",
                    trendline="ols",
                    title=f"Macro vs Micro: {self._wrap_text(macro_var, 28)} vs {self._wrap_text(target_col, 36)} (2019)",
                )
                fig.write_json(
                    self.eda_out / f"scatter_{self._safe_filename(macro_var)}_vs_{self._safe_filename(target_col)}.json"
                )

        # Keep a stable canonical chart for backward compatibility in UI.
        electricity_col = [c for c in merged.columns if "electricity" in c]
        if electricity_col and "pmgsy_road_length_km" in merged.columns:
            canon_df = merged[["pmgsy_road_length_km", electricity_col[0], "district", "state"]].dropna()
            if not canon_df.empty:
                fig = px.scatter(
                    canon_df,
                    x="pmgsy_road_length_km",
                    y=electricity_col[0],
                    color="state",
                    hover_name="district",
                    trendline="ols",
                    title="Infrastructure vs Development: Road Length vs Electrification (2019)",
                    labels={
                        "pmgsy_road_length_km": "PMGSY Roads Built (km)",
                        electricity_col[0]: self._wrap_text(electricity_col[0], 40),
                    },
                )
                fig.write_json(self.eda_out / "scatter_pmgsy_vs_electricity.json")

    def run_pipeline(self):
        logger.info("--- Starting EDA Artifact Pipeline ---")
        self.load_data()
        self.generate_data_quality_reports()
        self.generate_correlation_matrix()
        self.generate_national_profiles()
        self.generate_temporal_drift()
        self.generate_spatial_maps_and_stats()
        self.generate_state_rankings()
        self.generate_macro_micro_scatter()
        logger.info(f"--- EDA Pipeline Complete. Artifacts saved to {self.eda_out} ---")

if __name__ == "__main__":
    engine = EDAArtifactEngine()
    engine.run_pipeline()