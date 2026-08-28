import argparse
import mlflow
import pandas as pd
from utils import Config

def export_experiment_metrics(exp_name: str, output_csv: str):
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    
    df = mlflow.search_runs(experiment_names=[exp_name])
    if df.empty:
      print(f"No runs found for experiment '{exp_name}'.")
      return

    # 1. Separate columns by category
    metric_cols = [c for c in df.columns if c.startswith("metrics.")]
    param_cols = [c for c in df.columns if c.startswith("params.")]
    meta_cols = ["run_id", "status", "start_time", "end_time"]
    
    cols_to_keep = [c for c in meta_cols if c in df.columns] + param_cols + metric_cols
    df_clean = df[cols_to_keep].copy()

    df_clean.columns = [
      c.replace("metrics.", "")
        .replace("params.", "")
      for c in df_clean.columns
    ]

    df_clean.to_csv(output_csv, index=False)
    print(f"Successfully exported {len(df_clean)} runs with {len(df_clean.columns)} clean columns to '{output_csv}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export clean MLflow metrics to CSV.")
    parser.add_argument("--exp_name", type=str, default="framework_validation")
    parser.add_argument("--output", type=str, default="benchmark_summary.csv")
    args = parser.parse_args()

    export_experiment_metrics(args.exp_name, args.output)