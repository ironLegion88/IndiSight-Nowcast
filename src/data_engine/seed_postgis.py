import os
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
        
        # Construct Database URI
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")
        db = os.getenv("POSTGRES_DB")
        
        self.db_uri = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        self.engine = create_engine(self.db_uri)

    def seed_spatial_data(self):
        """Pushes the LGD-mapped GeoJSON to PostGIS."""
        logger.info("Seeding Spatial Data (india_districts_lgd)...")
        try:
            gdf = gpd.read_file(self.processed_dir / "spatial/india_districts_lgd.geojson")
            
            # Reproject to standard WGS84 just in case
            if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            
            # Push to PostGIS using GeoAlchemy2
            gdf.to_postgis(
                name="dim_district_geom",
                con=self.engine,
                if_exists="replace",
                index=False,
                dtype={'geometry': Geometry(geometry_type='MULTIPOLYGON', srid=4326)}
            )
            logger.info("Successfully seeded 'dim_district_geom' table.")
        except Exception as e:
            logger.error(f"Failed to seed spatial data: {e}", exc_info=True)
            raise

    def seed_tabular_data(self):
        """Pushes the harmonized NFHS, PMGSY, and MGNREGA data to PostGIS."""
        logger.info("Seeding Tabular Data...")
        try:
            # 1. Seed NFHS (Long format for easy querying)
            nfhs_long = pd.read_parquet(self.processed_dir / "tabular/nfhs_pooled_long.parquet")
            nfhs_long.to_sql("fact_nfhs", con=self.engine, if_exists="replace", index=False)
            logger.info(f"Seeded 'fact_nfhs' with {len(nfhs_long)} records.")

            # 2. Seed PMGSY
            pmgsy = pd.read_parquet(self.processed_dir / "tabular/pmgsy_district_agg.parquet")
            pmgsy.to_sql("fact_pmgsy", con=self.engine, if_exists="replace", index=False)
            logger.info(f"Seeded 'fact_pmgsy' with {len(pmgsy)} records.")

            # 3. Seed MGNREGA
            mgnrega = pd.read_parquet(self.processed_dir / "tabular/mgnrega_state_clean.parquet")
            mgnrega.to_sql("fact_mgnrega", con=self.engine, if_exists="replace", index=False)
            logger.info(f"Seeded 'fact_mgnrega' with {len(mgnrega)} records.")
            
        except Exception as e:
            logger.error(f"Failed to seed tabular data: {e}", exc_info=True)
            raise

    def create_indexes(self):
        """Creates SQL indexes to speed up the LLM's Text-to-SQL queries."""
        logger.info("Creating database indexes...")
        with self.engine.connect() as conn:
            # We use text() from sqlalchemy to execute raw SQL
            from sqlalchemy import text
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_nfhs_dist_year ON fact_nfhs (district_lgd_code, year);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_nfhs_metric ON fact_nfhs (metric_name);"))
            conn.commit()
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