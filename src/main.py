from dotenv import load_dotenv
import os
import mlflow
import pandas as pd
import numpy as np
import uuid
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from utils import parse_args, load_config, set_global_seeds, setup_logger, set_mlflow_exp,Config

mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
mlflow.langchain.autolog()

from workflow.graph import build_graph
from workflow.schema import GraphState

def main():
  # ----- EXPERIMENT SETUP -----
  load_dotenv()
  args = parse_args()
  set_global_seeds(args.seed)
  logger = setup_logger(Config.LOG_DIR, args.exp_name)

  # MASTER RANDOM GENERATOR
  rng = np.random.default_rng(seed=args.seed)

  feature_map = load_config(args.mapping)

  set_mlflow_exp(args.exp_name, Config.MLFLOW_TRACKING_URI)

  app = build_graph(optimiser=args.optimiser)

  # Set up the RunnableConfig
  validation_llm = ChatOpenAI(
    # model="gpt-4o-mini", 
    model="gpt-5.4-mini", 
    temperature = 0,
    seed = args.seed)

  # ----- START THE RUN -----
  with mlflow.start_run(run_name=args.run_name) as run:
    mlflow.log_params(vars(args))

    config = RunnableConfig(
      # run_id=uuid.uuid4(),
      # configurable={"thread_id": str(uuid.uuid4())},
      metadata={
      "validation_llm": validation_llm,
      "exp_name": args.exp_name,
      "run_name": args.run_name,
      "rng": rng
    })

    initial_state = GraphState(
      n_pop=args.n_pop,
      s_prevalence=args.s_prevalence,
      y_prevalence=args.y_prevalence,
      diff_y_prev_factor=args.diff_y_prev_factor,
      target_raw_auprc=args.target_raw_auprc,
      target_disp=args.target_disp,
      target_biased_recall_disp=args.target_biased_recall_disp,
      target_biased_ppv_disp=args.target_biased_ppv_disp,
      disparity_tolerance=args.disparity_tolerance,
      feature_map=feature_map,
      max_retries=args.max_retries,
      seed=args.seed
    )

    final_state = app.invoke(initial_state, config)

    metrics_to_log = {
      "retry_count": int(final_state['retry_count']),
      "input_tokens": int(final_state['input_tokens']),
      "output_tokens": int(final_state['output_tokens'])
    }

    mlflow.log_metrics(metrics_to_log)

if __name__ == "__main__":
  main()