import json
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(module_name=__name__, log_sub_dir="utils")

class BenchmarkContract:
    def __init__(self, benchmarks_dir: str | Path = "data/processed/benchmarks"):
        self.benchmarks_dir = Path(benchmarks_dir)
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_run(self, run_dir: Path) -> dict:
        """Validates a single benchmark run directory against the expected contract."""
        issues = []
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "predictions.parquet"
        shap_path = run_dir / "shap_summary.json"
        
        # 1. Check Metrics JSON
        if not metrics_path.exists():
            issues.append("metrics.json is missing.")
        else:
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                if not isinstance(metrics, dict):
                    issues.append("metrics.json must contain a JSON object.")
                else:
                    required_metrics = {"R2", "MAE", "RMSE"}
                    missing_metrics = required_metrics - set(metrics.keys())
                    if missing_metrics:
                        issues.append(f"metrics.json missing keys: {missing_metrics}")
            except Exception as e:
                issues.append(f"Failed to parse metrics.json: {e}")

        # 2. Check Predictions Parquet
        if not predictions_path.exists():
            issues.append("predictions.parquet is missing.")
        else:
            try:
                df = pd.read_parquet(predictions_path)
                required_cols = {"district_lgd_code", "actual", "predicted", "delta"}
                missing_cols = required_cols - set(df.columns)
                if missing_cols:
                    issues.append(f"predictions.parquet missing columns: {missing_cols}")
            except Exception as e:
                issues.append(f"Failed to parse predictions.parquet: {e}")

        # 3. Check SHAP Summary JSON
        if not shap_path.exists():
            issues.append("shap_summary.json is missing.")
        else:
            try:
                with open(shap_path, "r") as f:
                    shap_data = json.load(f)
                if not isinstance(shap_data, dict):
                    issues.append("shap_summary.json must contain a JSON object.")
            except Exception as e:
                issues.append(f"Failed to parse shap_summary.json: {e}")
                
        return {
            "run_id": run_dir.name,
            "valid": len(issues) == 0,
            "issues": issues,
            "paths": {
                "metrics": metrics_path if metrics_path.exists() else None,
                "predictions": predictions_path if predictions_path.exists() else None,
                "shap": shap_path if shap_path.exists() else None
            }
        }
        
    def get_valid_benchmarks(self) -> list[dict]:
        """Scans the benchmarks directory and returns a list of valid benchmark runs."""
        valid_runs = []
        if not self.benchmarks_dir.exists():
            return valid_runs
            
        for run_dir in self.benchmarks_dir.iterdir():
            if run_dir.is_dir():
                validation = self.validate_run(run_dir)
                if validation["valid"]:
                    valid_runs.append(validation)
                else:
                    logger.warning(f"Benchmark run {run_dir.name} failed contract validation: {validation['issues']}")
                    
        return valid_runs

if __name__ == "__main__":
    validator = BenchmarkContract()
    valid_runs = validator.get_valid_benchmarks()
    print(f"Found {len(valid_runs)} valid benchmark runs.")
