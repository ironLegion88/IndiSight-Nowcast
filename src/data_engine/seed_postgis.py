import os
import time
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
from geoalchemy2 import Geometry
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import get_logger

# Load environment variables
load_dotenv()
logger = get_logger(module_name=__name__, log_sub_dir="database")

class PostGISSeeder:
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.retry_attempts = int(os.getenv("RETRY_ATTEMPTS", "3"))
        self.retry_backoff_seconds = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.0"))
        
        # Construct Database URI
        self._require_env_vars(["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB"])
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")
        db = os.getenv("POSTGRES_DB")
        
        self.db_uri = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        self.engine = create_engine(self.db_uri)

    @staticmethod
    def _require_env_vars(keys: list[str]) -> None:
        missing = [k for k in keys if not os.getenv(k)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

    @staticmethod
    def _require_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    def _retry(self, operation_name: str, fn):
        last_error = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                if attempt >= self.retry_attempts:
                    break
                sleep_seconds = self.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "%s failed (attempt %s/%s): %s. Retrying in %.1fs",
                    operation_name,
                    attempt,
                    self.retry_attempts,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        raise RuntimeError(f"{operation_name} failed after {self.retry_attempts} attempts") from last_error

    def seed_spatial_data(self):
        """Pushes the LGD-mapped GeoJSON to PostGIS."""
        logger.info("Seeding Spatial Data (india_districts_lgd)...")
        try:
            spatial_path = self.processed_dir / "spatial/india_districts_lgd.geojson"
            self._require_file(spatial_path)
            gdf = gpd.read_file(spatial_path)
            
            # Reproject to standard WGS84 just in case
            if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            
            # Push to PostGIS using GeoAlchemy2
            self._retry(
                "Seed dim_district_geom",
                lambda: gdf.to_postgis(
                    name="dim_district_geom",
                    con=self.engine,
                    if_exists="replace",
                    index=False,
                    dtype={'geometry': Geometry(geometry_type='MULTIPOLYGON', srid=4326)}
                ),
            )
            logger.info("Successfully seeded 'dim_district_geom' table.")
        except Exception as e:
            logger.error(f"Failed to seed spatial data: {e}", exc_info=True)
            raise

    def seed_tabular_data(self):
        """Pushes the harmonized NFHS, PMGSY, and MGNREGA data to PostGIS."""
        logger.info("Seeding Tabular Data...")
        try:
            nfhs_path = self.processed_dir / "tabular/nfhs_pooled_long.parquet"
            pmgsy_path = self.processed_dir / "tabular/pmgsy_district_agg.parquet"
            mgnrega_path = self.processed_dir / "tabular/mgnrega_state_clean.parquet"
            self._require_file(nfhs_path)
            self._require_file(pmgsy_path)
            self._require_file(mgnrega_path)

            # 1. Seed NFHS (Long format for easy querying)
            nfhs_long = pd.read_parquet(nfhs_path)
            self._retry(
                "Seed fact_nfhs",
                lambda: nfhs_long.to_sql("fact_nfhs", con=self.engine, if_exists="replace", index=False),
            )
            logger.info(f"Seeded 'fact_nfhs' with {len(nfhs_long)} records.")

            # 2. Seed PMGSY
            pmgsy = pd.read_parquet(pmgsy_path)
            self._retry(
                "Seed fact_pmgsy",
                lambda: pmgsy.to_sql("fact_pmgsy", con=self.engine, if_exists="replace", index=False),
            )
            logger.info(f"Seeded 'fact_pmgsy' with {len(pmgsy)} records.")

            # 3. Seed MGNREGA
            mgnrega = pd.read_parquet(mgnrega_path)
            self._retry(
                "Seed fact_mgnrega",
                lambda: mgnrega.to_sql("fact_mgnrega", con=self.engine, if_exists="replace", index=False),
            )
            logger.info(f"Seeded 'fact_mgnrega' with {len(mgnrega)} records.")
            
        except Exception as e:
            logger.error(f"Failed to seed tabular data: {e}", exc_info=True)
            raise

    def create_indexes(self):
        """Creates SQL indexes to speed up the LLM's Text-to-SQL queries."""
        logger.info("Creating database indexes...")
        def _create() -> None:
            with self.engine.connect() as conn:
                # We use text() from sqlalchemy to execute raw SQL
                from sqlalchemy import text
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_nfhs_dist_year ON fact_nfhs (district_lgd_code, year);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_nfhs_metric ON fact_nfhs (metric_name);"))
                conn.commit()

        self._retry("Create DB indexes", _create)
        logger.info("Database indexes created successfully.")

    def run(self):
        logger.info("--- Starting PostGIS Seeding Process ---")
        self.seed_spatial_data()
        self.seed_tabular_data()
        self.create_indexes()
        logger.info("--- PostGIS Seeding Complete ---")

if __name__ == "__main__":
    seeder = PostGISSeeder()
    seeder.run()