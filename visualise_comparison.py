import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_spectral_comparison():
    """
    Generates a side-by-side comparative visualization of ground truth HLS data 
    versus model-predicted surface reflectance for the Near-Infrared (NIR) band.
    """
    # --- Data Path Configuration ---
    # Note: Ensure these assets are present in the local directory before execution
    source_hls = 'state_0_district_0.tif'
    prediction_hls = 'IndiSight_Nowcast_PHYSICAL.tif'
    
    if not os.path.exists(source_hls) or not os.path.exists(prediction_hls):
        print("ERROR: Geospatial assets missing in local directory.")
        return

    # --- Data Acquisition ---
    with rasterio.open(source_hls) as src_orig:
        # Extracting NIR band (Band 4) using the validated 224x224 spatial window
        ground_truth_nir = src_orig.read(4, window=rasterio.windows.Window(2, 2, 224, 224))
        
    with rasterio.open(prediction_hls) as src_pred:
        reconstructed_nir = src_pred.read(4)

    # --- Visualization Synthesis ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Radiometric scaling consistent with HLS surface reflectance standards
    val_min, val_max = 0, 4000 
    
    # Ground Truth Visualization
    axes[0].imshow(ground_truth_nir, cmap='YlGn', vmin=val_min, vmax=val_max)
    axes[0].set_title('Ground Truth: NIR Band (HLS)', fontsize=12)
    axes[0].axis('off')
    
    # Model Reconstruction Visualization
    im_pred = axes[1].imshow(reconstructed_nir, cmap='YlGn', vmin=val_min, vmax=val_max)
    axes[1].set_title('IndiSight-Nowcast: NIR Reconstruction', fontsize=12)
    axes[1].axis('off')
    
    # Integrated Colorbar Configuration
    cbar = fig.colorbar(im_pred, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label('Surface Reflectance (Reflectance Units)', rotation=270, labelpad=15)
    
    plt.suptitle("Spatio-Spectral Fidelity Analysis: Prithvi-100M Reconstruction", fontsize=14, y=0.95)
    
    # Exporting high-resolution diagnostic plot
    output_plot = 'spectral_comparison_analysis.png'
    plt.savefig(output_plot, bbox_inches='tight', dpi=300)
    print(f"LOG: Diagnostic comparison exported to {output_plot}")
    plt.show()

if __name__ == "__main__":
    plot_spectral_comparison()