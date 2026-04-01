# Please run `pixi run -e gpu earthengine authenticate`, login with Google, generate the token, 
# and paste it (or auto-resolve) before running this script.
# Run the script with `pixi run -e gpu python src/data_engine/extract_gee.py`
# If the script successfully downloads 5 `.tif` files, remove the `.head(5)` and let the script run
# for the whole country

import ee
import geopandas as gpd
import requests
import zipfile
import io
import os
from tqdm import tqdm

def initialize_ee():
    """Initialize Earth Engine."""
    try:
        ee.Initialize()
        print("Earth Engine Initialized Successfully.")
    except Exception as e:
        print("Failed to initialize Earth Engine. Did you run 'earthengine authenticate'?")
        raise e

def download_district_patch(district_name, state_name, geometry, year=2023):
    """
    Downloads a 5km x 5km Sentinel-2 patch centered on the district.
    """
    # 1. Get the centroid of the district polygon
    centroid = geometry.centroid
    point = ee.Geometry.Point([centroid.x, centroid.y])
    
    # 2. Create a 2500m buffer around the point to get a 5x5 km box
    region = point.buffer(2500).bounds()
    
    # 3. Query Sentinel-2 Surface Reflectance
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(region)
                  .filterDate(f'{year}-01-01', f'{year}-12-31')
                  # Pre-filter for less cloudy images
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    
    if collection.size().getInfo() == 0:
        print(f"No images found for {district_name}")
        return False

    # 4. Create a median composite (removes temporary clouds/shadows)
    # Select RGB + Near Infrared bands
    image = collection.median().select(['B4', 'B3', 'B2', 'B8'])
    
    # 5. Get Download URL (Scale = 10m per pixel)
    try:
        url = image.getDownloadURL({
            'scale': 10,
            'crs': 'EPSG:4326',
            'region': region,
            'format': 'GEO_TIFF'
        })
        
        # 6. Download and extract the ZIP
        response = requests.get(url)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Extract the tif file
                for filename in z.namelist():
                    if filename.endswith('.tif'):
                        # Rename the file to the district name
                        safe_name = f"{state_name}_{district_name}".replace(" ", "_").replace("/", "")
                        out_path = os.path.join("data/processed/images", f"{safe_name}.tif")
                        
                        with open(out_path, 'wb') as f:
                            f.write(z.read(filename))
            return True
    except Exception as e:
        print(f"Error downloading {district_name}: {e}")
        return False

def main():
    initialize_ee()
    
    os.makedirs("data/processed/images", exist_ok=True)
    
    print("Loading district boundaries...")
    geojson_path = "data/raw/india_districts.geojson"
    if not os.path.exists(geojson_path):
        print("Please run get_boundaries.py first!")
        return
        
    gdf = gpd.read_file(geojson_path)
    
    # Once the script runs, remove the .head(5) to process all ~700 districts
    test_gdf = gdf.head(5)
    
    print(f"Starting extraction for {len(test_gdf)} districts...")
    for idx, row in tqdm(test_gdf.iterrows(), total=len(test_gdf)):
        # Adjust column names based on the actual GeoJSON properties
        # Mapshaper India districts usually use 'dt_name' and 'st_name'
        d_name = row.get('dt_name', f'district_{idx}')
        s_name = row.get('st_name', f'state_{idx}')
        
        download_district_patch(d_name, s_name, row['geometry'])

    print("Extraction complete. Check data/processed/images/")

if __name__ == "__main__":
    main()