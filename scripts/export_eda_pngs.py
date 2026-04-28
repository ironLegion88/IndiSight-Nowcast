import plotly.io as pio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "data" / "processed" / "eda_artifacts"
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Export ALL JSON artifacts from the directory (more comprehensive)
json_files = list(ARTIFACT_DIR.glob("*.json"))
ARTIFACTS = {f.name: f.stem + ".png" for f in json_files}

print(f"Found {len(ARTIFACTS)} JSON artifacts to export.")

failed = []
success = []
for fname, outname in ARTIFACTS.items():
    src = ARTIFACT_DIR / fname
    out = OUT_DIR / outname
    if not src.exists():
        print(f"Missing artifact: {src}")
        failed.append(fname)
        continue
    try:
        print(f"Loading {fname}...")
        fig = pio.from_json(src.read_text(encoding='utf-8'))
        # Use kaleido engine to write static image; allow larger scale for quality
        fig.write_image(str(out), engine="kaleido", scale=2)
        print(f"[OK] Wrote {outname}")
        success.append(fname)
    except Exception as e:
        print(f"[FAIL] Failed to export {fname}: {e}")
        failed.append(fname)

print(f"\nExport summary: {len(success)} succeeded, {len(failed)} failed.")
if failed:
    print("Failed exports:")
    for f in failed:
        print(f" - {f}")
