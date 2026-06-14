import os
import json
import copy
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from workflow.schema import GraphState, DatasetValidationResult
from generation.features import FEATURE_FORMULAS_CONTEXT, generate_clinical_ground_truth, generate_observed_features
from generation.bias import BIAS_FORMULAS_CONTEXT, apply_feature_bias
from generation.analysis import run_dataset_diagnostics, run_downstream_probe
from utils import Config as CoreConfig
from workflow.prompts import PipelinePrompts

def generate_ground_truth_data(state: GraphState, config: RunnableConfig) -> dict:
  """
  Executes the clinical ground truth simulation using the parameters stored in the GraphState, followed by feature generation, and saves the dataset as a CSV file.
  """
  print("---> Generating clinical ground truth and observed features")
  
  # Retrieve parameters from the state
  n_samples = state.n_samples
  s_prevalence = state.s_prevalence
  y_prevalence = state.y_prevalence
  feature_map = state.feature_map

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

  complete_df, updated_feature_map = generate_observed_features(
    ground_truth_df=ground_truth_df,
    feature_map=feature_map,
    rng=rng,
    trial_index=state.retry_count
  )
  
  return {
    "df": complete_df,
    "feature_map": updated_feature_map
  }

def apply_bias(state: GraphState, config: RunnableConfig) -> dict:
  print("---> Applying bias to the soc pathway")

  if state.df is None:
    raise ValueError("No active dataframe found in the graph state to apply bias. Ensure upstream generation succeeded.")

  metadata = config.get("metadata") or {}
  rng = metadata.get("rng")
  if rng is None:
    print("Warning: No active RNG stream found in RunnableConfig. Instantiating fallback stream.")
    rng = np.random.default_rng(seed=state.seed)

  biased_df, updated_feature_map = apply_feature_bias(
    df=state.df,
    feature_map=state.feature_map,
    trial_index=state.retry_count,
    rng=rng
  )

  return {
    "df": biased_df,
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
  Appends new trial's results to the running probe result string.
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
  
  new_probe_results_table = run_downstream_probe(
    df=state.df,
    feature_map=state.feature_map,
    output_dir=output_dir,
    rng=rng
  )

  new_results_str = (f"## Downstream Probe Results: Trial {state.retry_count} \n\n{new_probe_results_table}\n\n")

  accumulated_results = (state.probe_results or "") + new_results_str

  os.makedirs(output_dir, exist_ok=True)
  report_path = os.path.join(output_dir, "probe_results.md")
  with open(report_path, "w") as f:
    f.write(accumulated_results)
    
  print(f"Success! Downstream probe results written to: {report_path}")

  return {
    "probe_results": accumulated_results
  }

def validate_dataset(state: GraphState, config: RunnableConfig) -> dict:
  """
  LLM node: Inspects the original user query, the feature map parameters,
  the data summary (Table One), and downstream classifier probe results to validate 
  if the user's expectations and target disparities are satisfied.
  """
  current_trial = state.retry_count
  next_trial = current_trial + 1
  print(f"---> Validating generated dataset against user's expectations (Trial {current_trial})")

  if state.df is None:
    print("Error: No dataset found in state to validate.")
    return {"validation_passed": False}

  # Get the Table One
  metadata = config.get("metadata") or {}
  exp_name = metadata.get("exp_name", "default_exp")
  run_name = metadata.get("run_name", "default_run")
  output_dir = f"{CoreConfig.DATA_DIR}/{exp_name}/{run_name}"

  table_one_path = os.path.join(output_dir, "table_one.txt")
  table_one_content = "Table One artifact not found."
  if os.path.exists(table_one_path):
    with open(table_one_path, "r") as f:
      table_one_content = f.read()

  # User expectations
  user_query = state.messages[0].content

  # Feature map, including feature parameters
  feature_map_str = json.dumps(state.feature_map, indent=2)

  # Downstream probe results
  probe_results_str = state.probe_results or "No probe results available."
  
  llm = metadata["validation_llm"]
  structured_llm = llm.with_structured_output(DatasetValidationResult)

  prompt = ChatPromptTemplate.from_messages([
    ("system", PipelinePrompts.DATASET_VALIDATION_PROMPT),
    ("human", "Original User Query / Expectations:\n{query}")
  ])

  chain = prompt | structured_llm
  result: DatasetValidationResult = chain.invoke({
    "formulas_context": FEATURE_FORMULAS_CONTEXT,
    "bias_context": BIAS_FORMULAS_CONTEXT,
    "feature_map": feature_map_str,
    "table_one": table_one_content,
    "probe_results": probe_results_str,
    "current_trial": current_trial,
    "query": user_query
  })

  print(f"     [Evaluation Reasoning]: {result.reasoning}")
  print(f"     [Result Passed]: {result.is_acceptable}")

  updates: dict = {
    "validation_passed": result.is_acceptable,
  }
    
  if result.is_acceptable:
    updates["feature_map"] = state.feature_map
  else:
    updates["retry_count"] = state.retry_count + 1

    if result.adjusted_parameters:
      print(f"     [Optimisation]: Applying {len(result.adjusted_parameters)} fine-tuned overrides directly to feature_map.")
      
      updated_map = copy.deepcopy(state.feature_map)
      current_trial_key = f"parameters_trial_{current_trial}"
      next_trial_key = f"parameters_trial_{next_trial}"

      for pathway in updated_map:
          for feature in updated_map[pathway]:
            if current_trial_key in feature:
              feature[next_trial_key] = copy.deepcopy(feature[current_trial_key])
      
      for override in result.adjusted_parameters:
        pathway = override.pathway
        if pathway in updated_map:
          for feature in updated_map[pathway]:
            if feature.get("name") == override.name:
              if next_trial_key not in feature or feature[next_trial_key] is None:
                  feature[next_trial_key] = {}
              
              if override.gamma is not None:
                feature[next_trial_key]["gamma"] = override.gamma
              if override.beta is not None:
                feature[next_trial_key]["beta"] = override.beta
              if override.noise_std is not None:
                feature[next_trial_key]["noise_std"] = override.noise_std
              if override.absolute_thresholds is not None:
                feature[next_trial_key]["absolute_thresholds"] = override.absolute_thresholds
      
      updates["feature_map"] = updated_map

  return updates