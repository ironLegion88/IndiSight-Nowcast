# Please run `pixi run -e gpu earthengine authenticate`, login with Google, generate the token, 
# and paste it (or auto-resolve) before running this script.
# Run the script with `pixi run -e gpu python src/data_engine/extract_gee.py`

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
        # Explicitly declare your registered Cloud Project
        ee.Initialize(project='ee-shubhamagarwal1879')
        print("Earth Engine Initialized Successfully.")
    except Exception as e:
        print("Failed to initialize Earth Engine. Did you run 'earthengine authenticate'?")
        raise e

def download_district_patch(district_name, state_name, geometry, year=2023):
    """
    Downloads a 5km x 5km Sentinel-2 patch centered on the district.
    Configured for Prithvi-100M (6 Bands).
    """
    centroid = geometry.centroid
    point = ee.Geometry.Point([centroid.x, centroid.y])
    region = point.buffer(2500).bounds()
    
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(region)
                  .filterDate(f'{year}-01-01', f'{year}-12-31')
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    
    if collection.size().getInfo() == 0:
        print(f"No images found for {district_name}")
        return False

    image = collection.median().select(['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'])
    
    try:
        url = image.getDownloadURL({
            'scale': 10,
            'crs': 'EPSG:4326',
            'region': region,
            'format': 'GEO_TIFF'
        })
        
        response = requests.get(url)
        if response.status_code == 200:
            safe_name = f"{state_name}_{district_name}".replace(" ", "_").replace("/", "")
            out_path = os.path.join("data/processed/images", f"{safe_name}.tif")
            
            try:
                # Scenario A: Google sent a ZIP file
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    for filename in z.namelist():
                        if filename.endswith('.tif'):
                            with open(out_path, 'wb') as f:
                                f.write(z.read(filename))
            except zipfile.BadZipFile:
                # Scenario B: Google sent the raw .tif file directly
                with open(out_path, 'wb') as f:
                    f.write(response.content)
            
            return True
        else:
            print(f"Bad response for {district_name}: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error downloading {district_name}: {e}")
        return False

def main():
    initialize_ee()
    
    os.makedirs("data/processed/images", exist_ok=True)
    
    print("Loading district boundaries...")
    geojson_path = "data/raw/india_districts.geojson"
    if not os.path.exists(geojson_path):
        print(f"Error: Could not find {geojson_path}. Sarthak needs to run get_boundaries.py first!")
        return
        
    gdf = gpd.read_file(geojson_path)
    
    test_gdf = gdf
    
    print(f"Starting extraction for {len(test_gdf)} districts...")
    for idx, row in tqdm(test_gdf.iterrows(), total=len(test_gdf)):
        d_name = row.get('dt_name', f'district_{idx}')
        s_name = row.get('st_name', f'state_{idx}')
        
        download_district_patch(d_name, s_name, row['geometry'])

    print("Extraction complete. Check data/processed/images/")

if __name__ == "__main__":
    main()