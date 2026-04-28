import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

# Assuming Prithvi is implemented in your local Prithvi.py
from Prithvi import PrithviViT 

class IndiSightVisionEngine:
    def __init__(self, checkpoint_path, config_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize Prithvi-100M 
        # Prithvi typically expects (B, C, T, H, W)
        # For a single composite, T=1
        self.model = PrithviViT(config_path) 
        self.model.load_state_dict(torch.load(checkpoint_path, map_map=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Hook to capture the embedding from the last Transformer block
        self.embeddings = []

    def preprocess_image(self, tif_path):
        """
        Standardizes GEE TIFs for the model.
        GEE Median Composite -> Normalize -> Tensor
        """
        with rasterio.open(tif_path) as src:
            # Prithvi expects 6 bands: Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2
            data = src.read([1, 2, 3, 4, 5, 6]) 
            data = data.astype(np.float32)
            
            # Simple Min-Max or Z-score normalization based on Prithvi's pre-training
            data = (data - data.mean()) / (data.std() + 1e-6)
            
        # Add Batch and Temporal dimensions: [C, H, W] -> [1, C, 1, H, W]
        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(2)
        return tensor.to(self.device)

    @torch.no_grad()
    def extract_embedding(self, image_tensor):
        """
        Passes image through ViT and returns the global average pooled token.
        """
        # Forward pass through the encoder
        # We take the latent representation (usually the CLS token or mean of tokens)
        latent = self.model.forward_encoder(image_tensor) 
        
        # Average pooling across the spatial tokens to get a single vector
        embedding = torch.mean(latent, dim=1).cpu().numpy().flatten()
        return embedding

def run_pipeline(image_dir, output_path):
    engine = IndiSightVisionEngine(
        checkpoint_path="weights/prithvi_100M.pt", 
        config_path="configs/Prithvi_100M_config.yaml"
    )
    
    results = []
    image_paths = list(Path(image_dir).glob("*.tif"))

    for path in tqdm(image_paths, desc="Extracting Geospatial Embeddings"):
        # LGD Code is usually embedded in the filename: "502_Anantapur.tif"
        lgd_code = int(path.stem.split('_')[0])
        
        img_tensor = engine.preprocess_image(path)
        vec = engine.extract_embedding(img_tensor)
        
        results.append({
            "lgd_code": lgd_code,
            "embedding": vec
        })

    # Save as Parquet for the Tabular ML Layer
    df = pd.DataFrame(results)
    df.to_parquet(output_path)
    print(f"Vision Extraction Complete. Saved to {output_path}")