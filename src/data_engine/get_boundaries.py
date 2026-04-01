import geopandas as gpd
import os

def download_district_boundaries():
    """
    Downloads and cleans India district boundaries.
    """
    print("Fetching India District boundaries...")
    # Old link, no longer working
    # url = "https://raw.githubusercontent.com/datameet/maps/master/Districts/Census_2011/india_districts.geojson"
    url = "https://raw.githubusercontent.com/datameet/indian-district-boundaries/refs/heads/master/topojson/india-districts-727.json"
    # This one doesn't have CRS data
    # url = "https://raw.githubusercontent.com/datameet/indian-district-boundaries/refs/heads/master/topojson/india-districts-2019-734.json"
    
    try:
        districts = gpd.read_file(url)
        # Standardize column names to lowercase
        districts.columns = [c.lower() for c in districts.columns]
        
        output_path = "data/raw/india_districts.geojson"
        districts.to_file(output_path, driver='GeoJSON')
        print(f"Successfully saved {len(districts)} districts to {output_path}")
    except Exception as e:
        print(f"Error downloading boundaries: {e}")

if __name__ == "__main__":
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")
    download_district_boundaries()