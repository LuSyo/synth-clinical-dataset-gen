from typing import cast
from langgraph.graph import StateGraph, END, START
from workflow.schema import GraphState
from workflow.nodes import (
  generate_ground_truth_data, 
  save_dataset, 
  generate_diagnostics, 
  evaluate_downstream_probe, 
  validate_dataset
  )

def build_graph():
  workflow = StateGraph(GraphState)

  # Add nodes
  workflow.add_node("generate_ground_truth_data", generate_ground_truth_data)
  workflow.add_node("generate_diagnostics", generate_diagnostics)
  workflow.add_node("evaluate_downstream_probe", evaluate_downstream_probe)
  workflow.add_node("validate_dataset", validate_dataset)
  workflow.add_node("save_dataset", save_dataset)

  # Add edges
  workflow.add_edge(START, "generate_ground_truth_data")
  workflow.add_edge("generate_ground_truth_data", "generate_diagnostics")
  workflow.add_edge("generate_diagnostics", "evaluate_downstream_probe")
  workflow.add_edge("evaluate_downstream_probe", "validate_dataset")
  workflow.add_conditional_edges(
    "validate_dataset", 
    route_retry_generation,
    {
      "save_dataset": "save_dataset",
      "generate_ground_truth_data": "generate_ground_truth_data"
    })
  workflow.add_edge("save_dataset", END)

  return workflow.compile()

def route_retry_generation(state: GraphState) -> str:
  """
  Conditional router determining whether the pipeline terminates or loops back
  for another data regeneration step based on validation status and iteration budget.
  """
  if state.validation_passed:
    print("---> Validation Passed! Routing to saving phase.")
    return "save_dataset"

  if state.retry_count > state.max_retries:
    print(f"---> Validation Failed, but hard iteration ceiling reached ({state.retry_count - 1}/{state.max_retries}). Stopping loop.")
    return "save_dataset"

  print(f"---> Validation Failed. Iteration budget remaining ({state.retry_count}/{state.max_retries}). Routing back to regenerate data.")
  return "generate_ground_truth_data"
