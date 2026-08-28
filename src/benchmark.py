import os
import time
import copy
import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from utils import parse_args, load_config, set_global_seeds, Config, set_mlflow_exp
from workflow.graph import build_graph
from workflow.schema import GraphState
from generation.analysis import evaluate_downstream_transferability
from enums import TargetDisp

def run_benchmark():
  load_dotenv()
  args = parse_args()
  set_mlflow_exp(args.exp_name, Config.MLFLOW_TRACKING_URI)
  
  base_feature_map = load_config(args.mapping)
  df_configs = pd.read_csv(args.benchmark_csv)

  # Pre-build workflow graphs
  graphs = {opt: build_graph(optimiser=opt) for opt in args.optimisers_list}

  total_runs = len(df_configs) * len(args.seeds_list) * len(args.optimisers_list)
  run_idx = 0

  print(f"Starting benchmark suite: {total_runs} total executions.")

  for _, row in df_configs.iterrows():
    dataset_id = row["dataset"]
    target_disp_enum = TargetDisp(row["target disp"].strip().lower())
    
    for optimiser in args.optimisers_list:
      max_retries = args.max_retries_llm if optimiser == "llm" else args.max_retries_optuna

      for seed in args.seeds_list:
        run_idx += 1
        run_name = f"{optimiser}_{dataset_id}_{seed}"
        print(f"\n[{run_idx}/{total_runs}] Running: {run_name}")

        set_global_seeds(seed)
        rng = np.random.default_rng(seed=seed)

        # Initialize LLM only if needed
        validation_llm = None
        if optimiser == "llm":
          validation_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            seed=seed
          )

        with mlflow.start_run(run_name=run_name):
          mlflow.set_tags({
            "optimiser": optimiser,
            "dataset_id": dataset_id,
            "benchmark_suite": "ML4H"
          })
          mlflow.log_params({
            "dataset_id": dataset_id,
            "optimiser": optimiser,
            "seed": seed,
            "max_retries": max_retries,
            "y_prev": row["y prev"],
            "s_prev": row["s prev"],
            "y_prev_base_ratio": row["y prev base ratio"],
            "target_auprc": row["target auprc"],
            "target_disp": target_disp_enum.value,
            "target_recall_disp": row["target recall disp"],
            "target_ppv_disp": row["target ppv disp"],
            "disp_tolerance": row["disp tolerance"]
          })

          # 2. State & Config Setup
          config = RunnableConfig(metadata={
            "validation_llm": validation_llm,
            "exp_name": args.exp_name,
            "run_name": run_name,
            "rng": rng
          })

          initial_state = GraphState(
            n_pop=args.n_pop,
            s_prevalence=float(row["s prev"]),
            y_prevalence=float(row["y prev"]),
            diff_y_prev_factor=float(row["y prev base ratio"]),
            target_raw_auprc=float(row["target auprc"]),
            target_disp=target_disp_enum,
            target_biased_recall_disp=float(row["target recall disp"]),
            target_biased_ppv_disp=float(row["target ppv disp"]),
            disparity_tolerance=float(row["disp tolerance"]),
            feature_map=copy.deepcopy(base_feature_map),
            max_retries=max_retries,
            seed=seed
          )

          mlflow.log_metrics({
            "raw_gen_converged": 0,
            "bias_converged": 0
          })

          # 3. Execution with Wall-Clock Tracking & Fault Tolerance
          start_time = time.time()
          try:
            final_state = graphs[optimiser].invoke(initial_state, config)
            elapsed_time = time.time() - start_time

            mlflow.log_metrics({
              "retry_count": int(final_state["retry_count"]),
              "input_tokens": int(final_state.get("input_tokens", 0)),
              "output_tokens": int(final_state.get("output_tokens", 0)),
              "wall_clock_sec": elapsed_time
            })
            print(f"---> Generation completed ({elapsed_time:.1f}s)")

            if final_state.get("df") is not None:
              print("---> Evaluating Downstream Performance and Disparity (LR, RF, MLP)")
              downstream_metrics = evaluate_downstream_transferability(
                df=final_state['df'],
                n_train=args.n_train,
                n_test=args.n_test,
                n_boot=args.n_boot,
                feature_map=base_feature_map,
                rng=rng
              )
              mlflow.log_metrics(downstream_metrics)
              print(f"---> Donwstream evaluation completed")

          except Exception as e:
            print(f"--> [ERROR] Run failed: {str(e)}")
            mlflow.set_tag("run_status", "failed")
            mlflow.log_param("error_message", str(e))

if __name__ == "__main__":
  run_benchmark()