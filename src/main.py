from dotenv import load_dotenv
import os
import mlflow
import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from utils import parse_args, load_config, set_global_seeds, setup_logger, Config

mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
mlflow.langchain.autolog(
  log_traces=True, 
  run_tracer_inline=True
)

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

  result_dir = os.path.join(Config.RESULTS_DIR, args.exp_name, args.run_name)
  os.makedirs(result_dir, exist_ok=True)

  feature_map = load_config(args.mapping)

  mlflow.set_experiment(args.exp_name)

  mlflow.langchain.autolog()

  app = build_graph()

  # Set up the RunnableConfig
  validation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=args.seed)
  config = RunnableConfig(metadata={
    "validation_llm": validation_llm,
    "exp_name": args.exp_name,
    "run_name": args.run_name,
    "rng": rng
  })

  # ----- START THE RUN -----
  with mlflow.start_run(run_name=args.run_name) as run:
    mlflow.log_params(vars(args))

    initial_state = GraphState(
      messages=[HumanMessage(content=args.query)],
      n_pop=args.n_pop,
      s_prevalence=args.s_prevalence,
      y_prevalence=args.y_prevalence,
      feature_map=feature_map,
      max_retries=args.max_retries,
      seed=args.seed
    )

    logger.info(f"QUERY: {args.query}")

    app.invoke(initial_state, config)

if __name__ == "__main__":
  main()