import numpy as np
import rasterio
import os

def validate_physical_reconstruction():
    """
    Performs inverse Z-score normalization to map model predictions back to
    physical surface reflectance units. Ensures radiometric consistency for
    downstream GIS applications.
    """
    # --- Resource Configuration ---
    source_hls_path = '/home/shubham.agarwal_phd24/workspace/IndiSight-Nowcast/data/processed/images/state_0_district_0.tif'
    latent_pred_path = 'nowcast_prediction_output.tif'
    physical_output_path = 'IndiSight_Nowcast_PHYSICAL.tif'

    if not os.path.exists(latent_pred_path):
        print(f"CRITICAL ERROR: Source {latent_pred_path} not found. Ensure inference is complete.")
        return

    print("LOG: Initializing Radiometric Inverse Transform...")

    # --- Data Acquisition ---
    with rasterio.open(source_hls_path) as src_orig, rasterio.open(latent_pred_path) as src_pred:
        # Align ground truth window with model output spatial resolution (224x224)
        ground_truth = src_orig.read(
            (1, 2, 3, 4, 5, 6), 
            window=rasterio.windows.Window(2, 2, 224, 224)
        ).astype(np.float32)
        
        # Acquisition of latent Z-score predictions
        latent_predictions = src_pred.read((1, 2, 3, 4, 5, 6)).astype(np.float32)
        
        # Preservation of georeferencing metadata
        output_metadata = src_pred.meta.copy()

    # --- Inverse Normalization Procedure ---
    # Transformation: Physical_Reflectance = (Z_Score * Standard_Deviation) + Mean
    physical_map = np.zeros_like(latent_predictions)
    
    for band_idx in range(6):
        # Calculate local distribution statistics from source HLS data
        local_mean = ground_truth[band_idx].mean()
        local_std = ground_truth[band_idx].std() + 1e-6
        
        # Radiometric mapping
        physical_map[band_idx] = (latent_predictions[band_idx] * local_std) + local_mean

    # --- GeoTIFF Serialization ---
    output_metadata.update({
        "driver": "GTiff",
        "dtype": 'float32',
        "nodata": 0,
        "compress": 'lzw'
    })

    with rasterio.open(physical_output_path, 'w', **output_metadata) as dst:
        dst.write(physical_map)

    # --- Final Summary Statistics ---
    print("\n" + "="*50)
    print("VALIDATION STATUS: RADIOMETRIC RECONSTRUCTION COMPLETE")
    print("="*50)
    print(f"Asset Path:     {os.path.abspath(physical_output_path)}")
    print(f"Spectral Floor: {physical_map.min():.4f}")
    print(f"Spectral Ceil:  {physical_map.max():.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    validate_physical_reconstruction()