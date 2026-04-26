# IndiSight-Nowcast: Spatio-Temporal District Reconstruction

IndiSight-Nowcast is a geospatial inference framework utilizing the Prithvi-100M Vision Transformer (ViT) backbone. The system is specifically optimized for high-fidelity nowcasting of multispectral HLS data, focusing on Indian regional topography and agricultural landscapes.

## Technical Contributions
- **Temporal Sequence Modeling:** Implements a 3-frame temporal stacking approach (T=3) to leverage the 3D Tubelet Attention mechanism for dynamic feature extraction.
- **Interleaved Spatial Decoding:** Solves the spatial reconstruction challenge by implementing a custom unpatchifying logic, mapping latent transformer tokens (14 x 14 grid) back to continuous 6-band multispectral maps.
- **Radiometric Validation:** Employs an automated inverse-transform pipeline to project model-normalized outputs back into physical surface reflectance units (0 - 4000).
- **Adaptive Normalization:** Utilizes local Z-score scaling to handle regional spectral variance across diverse Indian districts.

## Performance Analysis
The system achieved an aggregate spectral correlation of 0.4577 across all 6 HLS bands. The model demonstrates significant fidelity in capturing biophysical land-cover characteristics, particularly within the NIR and SWIR-1 spectral ranges.

## Repository Structure
| File | Description |
| :--- | :--- |
| Prithvi.py | Core Spatio-Temporal ViT architecture definition. |
| inference_nowcast.py | Main inference engine for temporal nowcasting. |
| final_validate.py | Radiometric scaling and physical unit mapping script. |
| visualise_comparison.py | Diagnostic utility for spectral-spatial fidelity analysis. |
| Prithvi_100M_config.yaml | Model hyperparameters and training statistics. |

## Execution Pipeline
1. **Inference:** Execute inference_nowcast.py to generate latent predictions from source HLS assets.
2. **Post-Processing:** Run final_validate.py to perform the radiometric inverse-transform and export a georeferenced GeoTIFF.
3. **Evaluation:** Utilize visualise_comparison.py for comparative spectral analysis.
