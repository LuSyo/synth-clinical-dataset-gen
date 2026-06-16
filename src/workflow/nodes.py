import os
import json
import copy
import numpy as np
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import create_model, ConfigDict
from workflow.schema import AccessBarrierParams, GraphState, DatasetValidationResult, ValidationResult, MeasurementErrorParams, ReferralBiasParams, UnderClassificationParams
from generation.features import FEATURE_FORMULAS_CONTEXT, generate_clinical_ground_truth, generate_observed_features
from generation.bias import BIAS_FORMULAS_CONTEXT, apply_feature_bias
from generation.analysis import plot_dataset, create_table_one, run_downstream_probe
from generation.processing import stratified_sampling
from utils import Config as CoreConfig
from workflow.prompts import PipelinePrompts

def generate_ground_truth_data(state: GraphState, config: RunnableConfig) -> dict:
  """
  Executes the clinical ground truth simulation using the parameters stored in the GraphState, followed by feature generation, and saves the dataset as a CSV file.
  """
  print("---> Generating clinical ground truth and observed features")
  
  # Retrieve parameters from the state
  n_pop = state.n_pop
  s_prevalence = state.s_prevalence
  y_prevalence = state.y_prevalence
  feature_map = state.feature_map

  metadata = config.get("metadata") or {}
  rng = metadata.get("rng")
  if rng is None:
    print("Warning: No active RNG stream found in RunnableConfig. Instantiating fallback stream.")
    rng = np.random.default_rng(seed=state.seed)
  
  ground_truth_df = generate_clinical_ground_truth(
    n_pop=n_pop,
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

  # Save population dataset
  if state.df is not None:
    pop_path = os.path.join(output_dir, "population.csv")
    state.df.to_csv(pop_path, index=False)
    print(f"  [Artifact Saved] Total Population -> {pop_path}")

  if state.train_df is not None:
    train_path = os.path.join(output_dir, "train.csv")
    state.train_df.to_csv(train_path, index=False)
    print(f"  [Artifact Saved] Training Set    -> {train_path}")

  if state.test_df is not None:
    test_path = os.path.join(output_dir, "test.csv")
    state.test_df.to_csv(test_path, index=False)
    print(f"  [Artifact Saved] Testing Set     -> {test_path}")

  # Save feature map
  map_output_path = os.path.join(output_dir, "feature_map.json")
  with open(map_output_path, "w") as f:
    json.dump(state.feature_map, f, indent=2)
  print(f"Success! Saved feature map to: {map_output_path}")
  
  return {}

def generate_plots(state: GraphState, config: RunnableConfig) -> dict:
  print("---> Generating Dataset Plots")

  if state.df is None:
    raise ValueError("No dataset found in the graph state to analyze. Ensure the generation node ran successfully.")

  metadata = config.get("metadata") or {}
  exp_name = metadata.get("exp_name", "default_exp")
  run_name = metadata.get("run_name", "default_run")
  
  output_dir = f"{CoreConfig.DATA_DIR}/{exp_name}/{run_name}"
  os.makedirs(output_dir, exist_ok=True)

  plot_dataset(df=state.df, feature_map=state.feature_map, output_dir=output_dir)
  print(f"Plots rendered and saved in: {output_dir}")

  return {}

def generate_table_one(state: GraphState, config: RunnableConfig) -> dict:
  if state.df is None:
    raise ValueError("No dataset found in the graph state to analyze. Ensure the generation node ran successfully.")

  metadata = config.get("metadata") or {}
  exp_name = metadata.get("exp_name", "default_exp")
  run_name = metadata.get("run_name", "default_run")
  
  output_dir = f"{CoreConfig.DATA_DIR}/{exp_name}/{run_name}"
  os.makedirs(output_dir, exist_ok=True)

  create_table_one(df=state.df, groupby="S", feature_map=state.feature_map, output_dir=output_dir)
  print(f"Table One saved in: {output_dir}")

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

  n_train = metadata.get("n_train", CoreConfig.N_TRAIN)
  n_test = metadata.get("n_test", CoreConfig.N_TEST)
  
  new_probe_results_table = run_downstream_probe(
    df=state.df,
    feature_map=state.feature_map,
    n_test=n_test,
    n_train=n_train,
    current_phase=state.phase,
    rng=rng
  )

  heading_insert = "(Raw Baseline Features)" if state.phase == "generation" else "(Biased Observed Features)"
  new_results_str = (f"## Downstream Probe Results {heading_insert}: Trial {state.retry_count} \n\n{new_probe_results_table}\n\n")

  accumulated_results = (state.probe_results or "") + new_results_str

  os.makedirs(output_dir, exist_ok=True)
  report_path = os.path.join(output_dir, "probe_results.md")
  with open(report_path, "w") as f:
    f.write(accumulated_results)
    
  print(f"Success! Downstream probe results written to: {report_path}")

  return {
    "probe_results": accumulated_results
  }

def sample_dataset(state: GraphState, config: RunnableConfig) -> dict:
  """
  Workflow node: Extracts runtime arguments from configuration and samples
  mutually exclusive training and testing splits from the main population.
  """
  print("---> Splitting population into training and testing subsets")

  if state.df is None:
    raise ValueError("No active dataframe found in the graph state to apply bias. Ensure upstream generation succeeded.")

  metadata = config.get("metadata") or {}
  rng = metadata.get("rng")
  if rng is None:
    print("Warning: No active RNG stream found in RunnableConfig. Instantiating fallback stream.")
    rng = np.random.default_rng(seed=state.seed)
  
  n_train = metadata.get("n_train", CoreConfig.N_TRAIN)
  n_test = metadata.get("n_test", CoreConfig.N_TEST)

  strata_columns = ['S']

  train_df, test_df = stratified_sampling(
    df=state.df,
    strata=strata_columns,
    n_train=n_train,
    n_test=n_test,
    rng=rng
  )

  return {
    "train_df": train_df,
    "test_df": test_df
  }

def validate_raw_dataset(state: GraphState, config: RunnableConfig) -> dict:
  """
  LLM node: Inspects the original user query, the feature map parameters,
  the data summary (Table One), and downstream classifier probe results to validate 
  if the user's expectations and target disparities are satisfied, and update feature generation parameters if not.
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
    ("system", PipelinePrompts.RAW_DATA_VALIDATION_PROMPT),
    ("human", "Original User Query / Expectations:\n{query}")
  ])

  chain = prompt | structured_llm
  result: DatasetValidationResult = chain.invoke({
    "formulas_context": FEATURE_FORMULAS_CONTEXT,
    "feature_map": feature_map_str,
    "table_one": table_one_content,
    "probe_results": probe_results_str,
    "current_trial": current_trial,
    "query": user_query
  })

  print(f"     [Raw dataset validation result]: {result.is_acceptable}")
  print(f"     [Evaluation Reasoning]: {result.reasoning}")

  updates: dict = {
    "validation_passed": result.is_acceptable,
  }
    
  if result.is_acceptable:
    updates["feature_map"] = state.feature_map
    updates["phase"] = "bias"
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

def validate_biased_dataset(state: GraphState, config: RunnableConfig) -> dict:
  """
  LLM node: Inspects the original user query, the feature map parameters,
  the data summary (Table One), and downstream classifier probe results to validate 
  if the user's expectations and target disparities are satisfied, and update feature bias parameters if not.
  """
  current_trial = state.retry_count
  next_trial = current_trial + 1
  print(f"---> Validating biased dataset against user's expectations (Trial {current_trial})")

  if state.df is None:
    print("Error: No dataset found in state to validate.")
    return {"validation_passed": False}

  # ======= Generate the schema ============
  BIAS_TYPE_MAP = {
    "measurement_error": MeasurementErrorParams,
    "access_barrier": AccessBarrierParams,
    "referral_bias": ReferralBiasParams,
    "under_classification": UnderClassificationParams
  }

  override_fields = {}

  for feature in state.feature_map.get("soc", []):
    f_name = feature.get("name")
    bias = feature.get("bias") or {}
    b_type = bias.get('type') or ""
    
    if b_type in BIAS_TYPE_MAP:
      override_fields[f_name] = (Optional[BIAS_TYPE_MAP[b_type]], None)

  DynamicBiasOverrides = create_model(
    "DynamicBiasOverrides",
    __config__= ConfigDict(extra="forbid"),
    **override_fields
  )

  DynamicBiasValidationResult = create_model(
    "DynamicBiasValidationResult",
    __base__=ValidationResult,
    __config__=ConfigDict(extra="forbid"),
    adjusted_parameters=(Optional[DynamicBiasOverrides], None)
  )

  # ==========================================

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
  structured_llm = llm.with_structured_output(DynamicBiasValidationResult)

  prompt = ChatPromptTemplate.from_messages([
    ("system", PipelinePrompts.BIASED_DATA_VALIDATION_PROMPT),
    ("human", "Original User Query / Expectations:\n{query}")
  ])

  chain = prompt | structured_llm
  result = chain.invoke({
    "formulas_context": FEATURE_FORMULAS_CONTEXT,
    "bias_context": BIAS_FORMULAS_CONTEXT,
    "feature_map": feature_map_str,
    "table_one": table_one_content,
    "probe_results": probe_results_str,
    "current_trial": current_trial,
    "query": user_query
  })

  print(f"     [Biased dataset validation result]: {result.is_acceptable}")
  print(f"     [Evaluation Reasoning]: {result.reasoning}")

  updates: dict = {
    "validation_passed": result.is_acceptable,
  }
    
  if result.is_acceptable:
    updates["feature_map"] = state.feature_map
    updates["phase"] = "complete"
  else:
    updates["retry_count"] = state.retry_count + 1

    updated_map = copy.deepcopy(state.feature_map)
    current_trial_key = f"parameters_trial_{current_trial}"
    next_trial_key = f"parameters_trial_{next_trial}"

    for feature in updated_map["soc"]:
      if current_trial_key in feature:
        feature[next_trial_key] = copy.deepcopy(feature[current_trial_key])

    if result.adjusted_parameters and "soc" in updated_map:
      print(f"       [Optimization]: Applying fine-tuned overrides to feature_map bias blocks.")

      emitted_overrides = result.adjusted_parameters.model_dump(exclude_none=True)

      for feature in updated_map["soc"]:
        f_name = feature.get("name")
        if f_name in emitted_overrides:
          if next_trial_key not in feature or feature[next_trial_key] is None:
            feature[next_trial_key] = {}

          if "bias_params" not in feature[next_trial_key] or feature[next_trial_key]["bias_params"] is None:
            feature[next_trial_key]["bias_params"] = {}
          
          for b_key, b_val in emitted_overrides[f_name].items():
            feature[next_trial_key]["bias_params"][b_key] = b_val
            print(f"         -> [{f_name}][{next_trial_key}] Set bias_params['{b_key}'] = {b_val}")

    updates["feature_map"] = updated_map

  return updates