from typing import cast
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
  validate_biased_dataset,
  validate_raw_dataset
  )

def build_graph():
  workflow = StateGraph(GraphState)

  # Add nodes
  workflow.add_node("generate_ground_truth_data", generate_ground_truth_data)
  workflow.add_node("validate_raw_dataset", validate_raw_dataset)
  workflow.add_node("apply_bias", apply_bias)
  workflow.add_node("generate_plots", generate_plots)
  workflow.add_node("generate_table_one_1", generate_table_one)
  workflow.add_node("generate_table_one_2", generate_table_one)
  workflow.add_node("evaluate_downstream_probe_1", evaluate_downstream_probe)
  workflow.add_node("evaluate_downstream_probe_2", evaluate_downstream_probe)
  workflow.add_node("validate_biased_dataset", validate_biased_dataset)
  workflow.add_node("sample_dataset", sample_dataset)
  workflow.add_node("save_dataset", save_dataset)

  # Add edges
  workflow.add_edge(START, "generate_ground_truth_data")
  workflow.add_edge("generate_ground_truth_data", "generate_table_one_1")
  workflow.add_edge("generate_table_one_1", "evaluate_downstream_probe_1")
  workflow.add_edge("evaluate_downstream_probe_1", "validate_raw_dataset")

  workflow.add_conditional_edges(
    "validate_raw_dataset", 
    route_retry_generation,
    {
      "generate_ground_truth_data": "generate_ground_truth_data",
      "apply_bias": "apply_bias"
    })

  workflow.add_edge("apply_bias", "generate_table_one_2")
  workflow.add_edge("generate_table_one_2", "evaluate_downstream_probe_2")

  workflow.add_conditional_edges(
    "evaluate_downstream_probe_2", 
    route_validate,
    {
      "sample_dataset": "sample_dataset",
      "validate_biased_dataset": "validate_biased_dataset"
    })
    
  workflow.add_conditional_edges(
    "validate_biased_dataset", 
    route_retry_bias,
    {
      "sample_dataset": "sample_dataset",
      "apply_bias": "apply_bias"
    })

  workflow.add_edge("sample_dataset", "generate_plots")
  workflow.add_edge("generate_plots", "save_dataset")
  workflow.add_edge("save_dataset", END)

  return workflow.compile()

def route_validate(state: GraphState) -> str:
  """
  Conditional router determining whether validation is needed to inform next retries
  or if max retry count has been reached
  """
  if state.retry_count >= state.max_retries:
    print(f"---> Hard iteration ceiling reached ({state.retry_count}/{state.max_retries}). Stopping loop.")
    return "sample_dataset"
  else:
    return "validate_biased_dataset"

def route_retry_generation(state: GraphState) -> str:
  """
  Conditional router determining whether the pipeline continues or loops back
  for another data regeneration step based on validation status and iteration budget.
  """
  if state.validation_passed:
    print("---> Validation Passed! Routing to bias application.")
    return "apply_bias"

  if state.retry_count >= state.max_retries // 2:
    print("---> Validation Failed. Saving retries budget for bias application.")
    return "apply_bias"

  print(f"---> Validation Failed. Iteration {state.retry_count} out of {state.max_retries}. Routing back to regenerate data.")
  return "generate_ground_truth_data"

def route_retry_bias(state: GraphState) -> str:
  """
  Conditional router determining whether the pipeline continues or loops back
  for another bias application step based on validation status.
  """
  if state.validation_passed:
    print("---> Validation Passed! Routing to bias application.")
    return "sample_dataset"

  print(f"---> Validation Failed. Iteration {state.retry_count} out of {state.max_retries}. Routing back to apply bias.")
  return "apply_bias"
