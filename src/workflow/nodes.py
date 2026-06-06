import os
import json
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from workflow.schema import GraphState, ExtractedDatasetParams, DatasetValidationResult
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

def validate_dataset(state: GraphState, config: RunnableConfig) -> dict:
  """
  LLM node: Inspects the original user query, the feature map parameters,
  the data summary (Table One), and downstream classifier probe results to validate 
  if the user's expectations and target disparities are satisfied.
  """
  print("---> Validating generated dataset against user's expectations")

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

  formulas_context = """
  ### Feature Generation Mechanics Context:
  - Clinical Ground Truth Latents:
    * H (Central health latent) ~ Gamma(shape=2.0, scale=1.5)
    * S (Sensitive attribute) ~ Bernoulli(s_prevalence)
    * U_dep (S-related latent) depends on H and S
      - If S == 0: latent_link = 0.75, noise_multiplier = 0.2
      - If S == 1: latent_link = 1.00, noise_multiplier = 3.0
      - U_dep ~ (latent_link * H) + noise_multiplier * Gamma(shape=1.2, scale=1)
    * U_indep (S-independent latent) depends on H only: 
      - U_indep ~ 0.6 * H + Gamma(shape=2.1, scale=1)
    * Y (Clinical outcome):
      - Y ~ Bernoulli(sigmoid(beta_0 + 1.5 * normalized(log(U_dep + U_indep))))
      - Note: beta_0 is calibrated via bisection search to exactly enforce the target y_prevalence.
  - Pathway Mappings to Latents:
    * "bio" features descend from latent U_dep
    * "soc" features descend from latent U_indep
    * "ind" features descend from latent U_indep
  - Observed Features (from Feature Map):
    * Continuous (Normal): X = gamma * Latent + beta + Normal(0, noise_std)
    * Continuous (Lognormal): X = exp(gamma * Latent + beta + Normal(0, noise_std))
    * Binary: P(X=1) = sigmoid(gamma * Latent + beta)
    * Categorical: Digitize an underlying continuous signal [gamma * Latent + Normal(0, noise_std)]
      - Note: The n-1 class boundaries are calculated between the 5th and 95th percentiles of the continuous signal.
  """

  llm = metadata["validation_llm"]
  structured_llm = llm.with_structured_output(DatasetValidationResult)

  prompt = ChatPromptTemplate.from_messages([
    ("system", (
      "You are an expert ML and Data Engineer assessing a synthetic dataset generation pipeline.\n"
      "Your objective is to verify if the generated dataset aligns with the user's approximate target "
      "predictive performance and group disparities as described in their original query.\n\n"
      "{formulas_context}\n\n"
      "### Current Feature Configuration (Feature Map):\n"
      "{feature_map}\n\n"
      "### Generated Dataset Summary (Table One):\n"
      "{table_one}\n\n"
      "### Downstream Classifier Performance & Disparities:\n"
      "{probe_results}\n\n"
      "Carefully analyze if the performance disparities or criteria specified in the user's query match "
      "the metrics shown above. Allow for slight stochastic variation. Provide your reasoning."
    )),
    ("human", "Original User Query / Expectations:\n{query}")
  ])

  chain = prompt | structured_llm
  result: DatasetValidationResult = chain.invoke({
    "formulas_context": formulas_context,
    "feature_map": feature_map_str,
    "table_one": table_one_content,
    "probe_results": probe_results_str,
    "query": user_query
  })

  print(f"     [Evaluation Reasoning]: {result.reasoning}")
  print(f"     [Result Passed]: {result.is_acceptable}")

  next_retry_count = state.retry_count if result.is_acceptable else state.retry_count + 1

  return {
    "validation_passed": result.is_acceptable,
    "retry_count": next_retry_count
  }