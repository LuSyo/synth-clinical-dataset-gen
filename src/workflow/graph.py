from typing import cast
from langgraph.graph import StateGraph, END, START
from workflow.schema import GraphState
from workflow.nodes import generate_ground_truth_data, save_dataset, generate_diagnostics, evaluate_downstream_probe

def build_graph():
  workflow = StateGraph(GraphState)

  # Add nodes
  workflow.add_node("generate_ground_truth_data", generate_ground_truth_data)
  workflow.add_node("generate_diagnostics", generate_diagnostics)
  workflow.add_node("evaluate_downstream_probe", evaluate_downstream_probe)
  workflow.add_node("save_dataset", save_dataset)

  # Add edges
  workflow.add_edge(START, "generate_ground_truth_data")
  workflow.add_edge("generate_ground_truth_data", "generate_diagnostics")
  workflow.add_edge("generate_diagnostics", "evaluate_downstream_probe")
  workflow.add_edge("evaluate_downstream_probe", "save_dataset")
  workflow.add_edge("save_dataset", END)

  return workflow.compile()