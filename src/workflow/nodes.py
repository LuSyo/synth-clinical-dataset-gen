import os
import json
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from workflow.schema import GraphState, ExtractedDatasetParams
from generation.features import generate_clinical_ground_truth, generate_observed_features
from generation.analysis import run_dataset_diagnostics, run_downstream_probe
from utils import Config as CoreConfig

def extract_parameters(state: GraphState, config: RunnableConfig) -> dict:
  """
    Parses the user's query to retrieve the dataset parameters
  """
  print("---> Extracting dataset parameters")

  llm = config["metadata"]["generate_llm"] #type: ignore
  
  structured_llm = llm.with_structured_output(ExtractedDatasetParams)
  
  prompt = ChatPromptTemplate.from_messages([
    ("system", (
      "You are an expert clinical data analyst.\n"
      "Your task is to examine the user request and extract dataset parameters.\n"
      "Remember: the prevalence values MUST be converted to floats between 0 and 1.\n"
      "If a parameter is completely unmentioned, leave it as null."
    )),
    ("human", "{query}")
  ])
  
  user_query = state.messages[0].content
  
  chain = prompt | structured_llm
  extracted: ExtractedDatasetParams = chain.invoke({"query": user_query})
  
  updates = {}
  updates["n_samples"] = extracted.n_samples if extracted.n_samples else CoreConfig.N_SAMPLES
  updates["s_prevalence"] = extracted.s_prevalence if extracted.s_prevalence else CoreConfig.S_PREV
  updates["y_prevalence"] = extracted.y_prevalence if extracted.y_prevalence else CoreConfig.Y_PREV
      
  return updates

def generate_ground_truth_data(state: GraphState, config: RunnableConfig) -> dict:
  """
  Executes the clinical ground truth simulation using the parameters stored in the GraphState, followed by feature generation, and saves the dataset as a CSV file.
  """
  print("---> Generating clinical ground truth and observed features")
  
  # Retrieve parameters from the state
  n_samples = state.n_samples
  s_prevalence = state.s_prevalence
  y_prevalence = state.y_prevalence

  metadata = config.get("metadata") or {}
  rng = metadata.get("rng")
  if rng is None:
    print("Warning: No active RNG stream found in RunnableConfig. Instantiating fallback stream.")
    rng = np.random.default_rng(seed=state.seed)
  
  ground_truth_df = generate_clinical_ground_truth(
    n_samples=n_samples,
    s_prevalence=s_prevalence,
    y_prevalence=y_prevalence,
    rng=rng
  )

  feature_map = state.feature_map

  complete_df, updated_feature_map = generate_observed_features(
    ground_truth_df=ground_truth_df,
    feature_map=feature_map,
    rng=rng
  )
  
  return {
    "df": complete_df,
    "feature_map": updated_feature_map
  }

def save_dataset(state: GraphState, config: RunnableConfig) -> dict:
  print("---> Saving Generated Dataset to Disk")
    
  if state.df is None:
    raise ValueError("No dataset found in the graph state to save. Ensure the generation node ran successfully.")
  
  metadata = config.get("metadata") or {}
  exp_name = metadata.get("exp_name", "default_exp")
  run_name = metadata.get("run_name", "default_run")

  output_dir = f"{CoreConfig.DATA_DIR}/{exp_name}/{run_name}"
  os.makedirs(output_dir, exist_ok=True)

  # Save dataset
  dataset_output_path = os.path.join(output_dir, "clinical_ground_truth.csv")
  state.df.to_csv(dataset_output_path, index=False)
  print(f"Success! Saved dataset ({len(state.df)} rows) to: {dataset_output_path}")

  # Save feature map
  map_output_path = os.path.join(output_dir, "feature_map.json")
  with open(map_output_path, "w") as f:
    json.dump(state.feature_map, f, indent=2)
  print(f"Success! Saved feature map to: {map_output_path}")
  
  return {
    "dataset_path": dataset_output_path,
    "df": None
  }

def generate_diagnostics(state: GraphState, config: RunnableConfig) -> dict:
  print("---> Generating Dataset Diagnostics & Visualizations")

  if state.df is None:
    raise ValueError("No dataset found in the graph state to analyze. Ensure the generation node ran successfully.")

  metadata = config.get("metadata") or {}
  exp_name = metadata.get("exp_name", "default_exp")
  run_name = metadata.get("run_name", "default_run")
  
  output_dir = f"{CoreConfig.DATA_DIR}/{exp_name}/{run_name}"
  os.makedirs(output_dir, exist_ok=True)

  run_dataset_diagnostics(df=state.df, feature_map=state.feature_map, output_dir=output_dir)
  print(f"Diagnostics logs and plots compiled in: {output_dir}")

  return {}

def evaluate_downstream_probe(state: GraphState, config: RunnableConfig) -> dict:
  """
  Workflow evaluation node: Executes the 4-iteration bootstrap classifier 
  probe on the active dataframe to measure baseline performance and disparities.
  """
  if state.df is None:
    raise ValueError("No dataset found in the graph state to probe. Ensure upstream generation nodes succeeded.")

  metadata = config.get("metadata") or {}
  exp_name = metadata.get("exp_name", "default_exp")
  run_name = metadata.get("run_name", "default_run")
  
  rng = metadata.get("rng")
  if rng is None:
    rng = np.random.default_rng(seed=state.seed)
    
  output_dir = f"{CoreConfig.DATA_DIR}/{exp_name}/{run_name}"
  
  probe_results = run_downstream_probe(
    df=state.df,
    feature_map=state.feature_map,
    output_dir=output_dir,
    rng=rng
  )

  return {
    "probe_results": probe_results
  }

# def validate_dataset(state: GraphState, config: RunnableConfig) -> dict:
#   print("---> Validating generated dataset")

#   if state.df is None:
#     print("Error: No dataset found in state.")
#     return {"validation_status": "generation_failure"}

#   dataset = state.df

#   actual_n = len(dataset)
#   actual_s_prev = (dataset['S'] == 1).mean()
#   actual_y_prev = (dataset['Y'] == 1).mean()

#   summary_msg = (
#     f"Target vs Actual Metrics:\n"
#     f" - Samples (N): Target={state.n_samples} | Actual={actual_n}\n"
#     f" - Majority Prevalence (S=1): Target={state.s_prevalence} | Actual={actual_s_prev:.4f}\n"
#     f" - Outcome Prevalence (Y=1): Target={state.y_prevalence} | Actual={actual_y_prev:.4f}"
#   )
  
#   original_user_query = state.messages[0].content

#   return {}