from typing import cast
import mlflow
from langgraph.graph import StateGraph, END, START
from workflow.schema import GraphState
from workflow.nodes import (
  generate_ground_truth_data, 
  apply_bias,
  save_dataset, 
  generate_plots, 
  generate_table_one,
  evaluate_downstream_probe, 
  sample_dataset,
  reparametrise_biased_dataset,
  reparametrise_raw_dataset,
  reparametrise_raw_dataset_optuna,
  reparametrise_biased_dataset_optuna
  )

def build_graph(optimiser: str = "llm"):
  workflow = StateGraph(GraphState)

  # Add nodes
  workflow.add_node("generate_ground_truth_data", generate_ground_truth_data)
  workflow.add_node("apply_bias", apply_bias)
  workflow.add_node("generate_plots", generate_plots)
  workflow.add_node("generate_table_one_1", generate_table_one)
  workflow.add_node("generate_table_one_2", generate_table_one)
  workflow.add_node("evaluate_downstream_probe_1", evaluate_downstream_probe)
  workflow.add_node("evaluate_downstream_probe_2", evaluate_downstream_probe)
  workflow.add_node("sample_dataset", sample_dataset)
  workflow.add_node("save_dataset", save_dataset)

  if optimiser == "llm":
    workflow.add_node("reparametrise_raw_dataset", reparametrise_raw_dataset)
    workflow.add_node("reparametrise_biased_dataset", reparametrise_biased_dataset)
  else:
    workflow.add_node("reparametrise_raw_dataset", reparametrise_raw_dataset_optuna)
    workflow.add_node("reparametrise_biased_dataset", reparametrise_biased_dataset_optuna)

  # Add edges
  workflow.add_edge(START, "generate_ground_truth_data")
  workflow.add_edge("generate_ground_truth_data", "generate_table_one_1")
  workflow.add_edge("generate_table_one_1", "evaluate_downstream_probe_1")

  workflow.add_conditional_edges(
    "evaluate_downstream_probe_1", 
    route_reparametrise,
    {
      "skip": "apply_bias",
      "reparametrise": "reparametrise_raw_dataset"
    })
  
  workflow.add_edge("reparametrise_raw_dataset", "generate_ground_truth_data")

  workflow.add_edge("apply_bias", "generate_table_one_2")
  workflow.add_edge("generate_table_one_2", "evaluate_downstream_probe_2")

  workflow.add_conditional_edges(
    "evaluate_downstream_probe_2", 
    route_reparametrise,
    {
      "skip": "sample_dataset",
      "reparametrise": "reparametrise_biased_dataset"
    })

  workflow.add_edge("reparametrise_biased_dataset", "apply_bias")

  workflow.add_edge("sample_dataset", "generate_plots")
  workflow.add_edge("generate_plots", "save_dataset")
  workflow.add_edge("save_dataset", END)

  return workflow.compile()

def route_reparametrise(state: GraphState) -> str:
  """
  Conditional router determining whether validation is needed to inform next retries
  or if max retry count has been reached
  """
  if state.validation_passed:
    print("---> Validation passed. Moving to next phase.")
    if state.phase == "generation":
      mlflow.log_metrics({
        "raw_gen_trials": state.retry_count,
        "raw_gen_converged": 1
      })
    else:
      mlflow.log_metrics({
        "bias_converged": 1
      })
    return "skip"

  elif state.phase == "generation" and state.retry_count >= state.max_retries // 2:
    print("---> Skipping reparametrisation: Saving retries budget for bias application.")
    mlflow.log_metrics({"raw_gen_trials": state.retry_count})
    return "skip"
    
  elif state.retry_count >= state.max_retries:
    print(f"---> Hard iteration ceiling reached ({state.retry_count}/{state.max_retries}). Stopping reparametrisation loop.")
    return "skip"

  
  else:
    return "reparametrise"