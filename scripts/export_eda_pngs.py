import plotly.io as pio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "data" / "processed" / "eda_artifacts"
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# List of artifacts to export (artifact filename -> output basename)
ARTIFACTS = {
    "correlation_heatmap.json": "correlation_heatmap.png",
    "map_hh_electricity.json": "map_hh_electricity.png",
    "distribution_hh_electricity.json": "distribution_hh_electricity.png",
    "drift_box_hh_electricity.json": "drift_box_hh_electricity.png",
    "trend_hh_electricity.json": "trend_hh_electricity.png",
    "ranking_hh_electricity.json": "ranking_hh_electricity.png",
    "scatter_pmgsy_vs_electricity.json": "scatter_pmgsy_vs_electricity.png",
}

failed = []
for fname, outname in ARTIFACTS.items():
    src = ARTIFACT_DIR / fname
    out = OUT_DIR / outname
    if not src.exists():
        print(f"Missing artifact: {src}")
        failed.append(fname)
        continue
    try:
        print(f"Loading {src}...")
        fig = pio.from_json(src.read_text(encoding='utf-8'))
        # Use kaleido engine to write static image; allow larger scale for quality
        fig.write_image(str(out), engine="kaleido", scale=2)
        print(f"Wrote {out}")
    except Exception as e:
        print(f"Failed to export {src}: {e}")
        failed.append(fname)

if failed:
    print("Export completed with failures:")
    for f in failed:
        print(f" - {f}")
else:
    print("All artifacts exported successfully.")
