import torch
import yaml
import rasterio
import numpy as np
import os
from Prithvi import MaskedAutoencoderViT
from rasterio.windows import Window

def main():
    """
    Main execution script for temporal nowcasting inference using Prithvi-100M.
    Handles data ingestion, local normalization, and spatial-temporal reconstruction.
    """
    # --- Configuration and Environment Setup ---
    # Path to the source HLS multi-spectral image
    file_path = '/home/shubham.agarwal_phd24/workspace/IndiSight-Nowcast/data/processed/images/state_0_district_0.tif'
    config_path = "Prithvi_100M_config.yaml"
    weights_path = "Prithvi_100M.pt"
    device = torch.device("cpu") 

    # --- Model Initialization ---
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model_args = config["model_args"]
    model_args["num_frames"] = 3
    model = MaskedAutoencoderViT(**model_args)
    
    # --- Weight Synchronization ---
    # Adjusting state_dict keys to align with local model architecture
    checkpoint = torch.load(weights_path, map_location="cpu")
    new_state_dict = {}
    for key, value in checkpoint.items():
        new_key = key.replace("encoder.", "").replace("decoder.", "").replace("patch_embed.proj", "patch_embed")
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    # --- Data Ingestion ---
    print("LOG: Accessing source multispectral data...")
    with rasterio.open(file_path) as src:
        # Utilizing a 226x226 window to facilitate temporal frame extraction (T-1, T, T+1)
        raw_data = src.read((1,2,3,4,5,6), window=Window(0,0,226,226)).astype(np.float32)
        src_meta = src.meta.copy()
        src_transform = src.transform
        
    # --- Temporal Stack Generation ---
    # Creating a 3-frame sequence for the transformer's temporal attention mechanism
    frame_t_minus_1 = raw_data[:, 0:224, 0:224]
    frame_t = raw_data[:, 1:225, 1:225]
    frame_t_plus_1 = raw_data[:, 2:226, 2:226]
    temporal_stack = np.stack([frame_t_minus_1, frame_t, frame_t_plus_1], axis=1)

    # --- Local Adaptive Normalization ---
    # Implementing band-wise Z-score normalization for regional spectral adaptation
    for i in range(6):
        band_mean = temporal_stack[i].mean()
        band_std = temporal_stack[i].std() + 1e-6
        temporal_stack[i] = (temporal_stack[i] - band_mean) / band_std

    input_tensor = torch.from_numpy(temporal_stack).unsqueeze(0).to(device).float()

    # --- Model Inference ---
    print("LOG: Executing Spatio-Temporal Nowcast Inference...")
    with torch.no_grad():
        _, prediction, _ = model(input_tensor)
    
    # --- Spatial Decoding (Unpatchify) ---
    # Mapping 1D tokens back to a 6-band interleaved spatial grid
    future_tokens = prediction[:, 393:589, :] 
    
    # Reshape logic: [Batch, Patch_Grid_H, Patch_Grid_W, Pixel_H, Pixel_W, Bands]
    patch_reshaped = future_tokens.reshape(1, 14, 14, 16, 16, 6) 
    
    # Permute to standard raster format: [Batch, Bands, Patch_H, Pixel_H, Patch_W, Pixel_W]
    patch_permuted = patch_reshaped.permute(0, 5, 1, 3, 2, 4) 
    future_map = patch_permuted.reshape(6, 224, 224).cpu().numpy()
    
    # --- Statistical Validation ---
    print("\n" + "="*45)
    print("METRICS: PER-BAND SPECTRAL CORRELATION")
    print("-" * 45)
    total_correlation = 0
    for band in range(6):
        correlation = np.corrcoef(frame_t_plus_1[band].flatten(), future_map[band].flatten())[0, 1]
        total_correlation += correlation
        print(f"Band {band+1} Coefficient: {correlation:.4f}")
    
    mean_correlation = total_correlation / 6
    print("-" * 45)
    print(f"Aggregate Temporal Correlation: {mean_correlation:.4f}")
    print("="*45 + "\n")

    # --- Output Export ---
    src_meta.update({
        "driver": "GTiff",
        "height": 224,
        "width": 224,
        "count": 6,
        "dtype": 'float32',
        "transform": rasterio.windows.transform(Window(2, 2, 224, 224), src_transform)
    })

    output_path = 'nowcast_prediction_output.tif'
    with rasterio.open(output_path, 'w', **src_meta) as dst:
        dst.write(future_map)
    
    print(f"SUCCESS: Reconstructed GeoTIFF exported to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()