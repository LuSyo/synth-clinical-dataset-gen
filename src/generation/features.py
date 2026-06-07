import numpy as np
import pandas as pd
import json
from typing import cast, Tuple, Dict, Any, Optional
from scipy.special import expit
from scipy.optimize import bisect

def generate_clinical_ground_truth(
  n_samples: int, 
  s_prevalence: float, 
  y_prevalence: float, 
  rng: np.random.Generator
) -> pd.DataFrame:
  """
  Generates the latent health state of the population, underlying cause of the outcome, as per the following SCM: 
  H -> U_corr
  H -> U_desc
  S -> U_corr
  U_corr -> Y
  U_desc -> Y

  Inputs:
    - n_samples (int): size of the population
    - s_prevalence (float, [0, 1]): prevalence of the majority group of the sensitive attribute S in the population
    - y_prevalence (float, [0, 1]): prevalence of the clinical outcome Y in the population
    - seed (int): random seed for reproducibility

  Outputs:
    - DataFrame with n_samples rows and the columns:
      - H: central health latent 
      - S: binary sensitive attribute
      - U_dep: Latent descending from H, directly influenced by S
      - U_indep: Latent descending from H, independent of S
  """
    
  # Central health latent H (Independent Gamma distribution)
  h_shape, h_scale = 2.0, 1.5
  H = rng.gamma(shape=h_shape, scale=h_scale, size=n_samples)
  
  # Binary Sensitive attribute S (Bernoulli/Binomial distribution)
  S = rng.binomial(n=1, p=s_prevalence, size=n_samples)
  
  # Latent U_indep (Descends from H, independent of S)
  U_indep = 0.6 * H + rng.gamma(shape=2.1, scale=1, size=n_samples)
  
  # Latent U_dep (Descends from H, directly influenced by S)
  latent_link = np.where(S == 0, 0.75, 1.0)
  multiplier = np.where(S == 0, 0.2, 3.0)
  U_dep = (latent_link * H) + multiplier * rng.gamma(shape=1.2, scale=1, size=n_samples)
  
  # Outcome
  structural_signal = np.log(U_dep + U_indep)
  structural_signal_normed = (structural_signal - structural_signal.mean())/structural_signal.std()

  # irreducible noise
  noise_factor = 1.5

  def objective_function(beta_0, scaled_signal, target):
    # Computes the difference between expected sample prevalence and target prevalence
    return np.mean(expit(beta_0 + scaled_signal)) - target

  # find the intercept where the objective function equals 0
  scaled_signal = noise_factor * structural_signal_normed
  beta_0 = bisect(objective_function, -10, 10, args=(scaled_signal, y_prevalence))

  # generate log-odds and probabilities
  log_odds_noisy = beta_0 + scaled_signal
  Y_prob = expit(log_odds_noisy)

  # stochastic sampling of the true outcome
  Y = rng.binomial(n=1, p=Y_prob)
  
  # Create the DataFrame matching the causal ground truth requirements
  df = pd.DataFrame({
      "H": H,
      "S": S,
      "U_dep": U_dep,
      "U_indep": U_indep,
      "Y": Y
  })
  
  return df

def generate_observed_features(ground_truth_df: pd.DataFrame,
    feature_map: dict,
    rng: np.random.Generator  
) -> Tuple[pd.DataFrame, dict]:
  """
    Generates the set of features from the pre-configured feature map and using the previously generated clinical ground truth
  """
  df = ground_truth_df.copy()
  enriched_map = json.loads(json.dumps(feature_map))

  pathway_latent_map = {
    "bio": "U_dep",
    "soc": "U_indep",
    "ind": "U_indep"
  }

  for pathway_key, feature_list in enriched_map.items():
    if pathway_key not in pathway_latent_map:
      print(f"Warning: Pathway '{pathway_key}' is not mapped to an SCM latent. Skipping.")
      continue
        
    parent_latent_name = pathway_latent_map[pathway_key]

    if parent_latent_name not in df.columns:
      print(f"Error: Latent '{parent_latent_name}' does not exist in the ground truth DataFrame.")

    latent_series = cast(pd.Series, df[parent_latent_name])
    
    for feature_spec in feature_list:
      name = feature_spec["name"]
      feat_type = feature_spec.get("type", "").lower()
      existing_params = feature_spec.get("parameters", None)
      
      if feat_type == "continuous":
        dist = feature_spec.get("dist", "normal")
        data_array, params = generate_continuous_feat(latent_series, dist, rng, existing_params)
          
      elif feat_type == "binary":
        data_array, params = generate_binary_feat(latent_series, rng, existing_params)
          
      elif feat_type == "categorical":
        n_classes = feature_spec.get("n", 3)
        data_array, params = generate_categorical_feat(latent_series, n_classes, rng, existing_params)
          
      else:
          raise ValueError(f"Unknown or unsupported feature type '{feat_type}' for feature '{name}'")

      df[name] = data_array
      feature_spec["parameters"] = params
              
  return df, enriched_map

def generate_continuous_feat(
  latent_series: pd.Series, 
  dist_type: str, 
  rng: np.random.Generator,
  existing_params: Optional[dict] = None
  ) -> Tuple[np.ndarray, Dict[str, Any]]:
  """
    Generates a continuous feature driven by a parent latent.
    Formula: X = gamma * Latent + beta + noise
  """

  n_samples = len(latent_series)

  if existing_params and all(k in existing_params for k in ["gamma", "beta", "noise_std"]):
    gamma = existing_params["gamma"]
    beta = existing_params["beta"]
    noise_std = existing_params["noise_std"]
  else:
    # RANDOM PARAMETERS
    gamma = float(rng.uniform(0.5, 2.0) * rng.choice([-1, 1]))  # Directional weight
    beta = float(rng.uniform(-1.0, 1.0))                        # Intercept shift
    if dist_type.lower().strip() == "lognormal":
      noise_std = float(rng.uniform(0.1, 0.4))
    else:
      noise_std = float(rng.uniform(0.5, 1.5))

  if dist_type.lower().strip() == "lognormal":
    underlying_normal = gamma * latent_series + beta + rng.normal(0, noise_std, size=n_samples)
    data = np.exp(underlying_normal)
    dist_type = "lognormal"
    
  else:  # Default to "normal"
    dist_type = "normal"
    data = gamma * latent_series + beta + rng.normal(0, noise_std, size=n_samples)

  params = {
    "gamma": gamma,
    "beta": beta,
    "noise_std": noise_std,
    "computed_distribution": dist_type
  }
  return data, params

def generate_binary_feat(
  latent_series: pd.Series, 
  rng: np.random.Generator,
  existing_params: Optional[dict] = None
  ) -> Tuple[np.ndarray, Dict[str, Any]]:
  """
    Generates a binary feature driven by a parent latent.
    Formula: P(X=1) = sigmoid(gamma * Latent + beta)
  """
  if existing_params and all(k in existing_params for k in ["gamma", "beta"]):
    gamma = existing_params["gamma"]
    beta = existing_params["beta"]
  else:
    gamma = float(rng.uniform(0.5, 1.8) * rng.choice([-1, 1]))
    beta = float(rng.uniform(-0.5, 0.5))
  
  probabilities = expit(gamma * latent_series + beta)
  data = rng.binomial(n=1, p=probabilities)

  params = {
    "gamma": gamma,
    "beta": beta
  }
  return data, params

def generate_categorical_feat(
  latent_series: pd.Series, 
  n_classes: int,
  rng: np.random.Generator,
  existing_params: Optional[dict] = None
  ) -> Tuple[np.ndarray, Dict[str, Any]]:
  """
    Generates an ordinal categorical feature using randomized absolute thresholds.
  """
  n_samples = len(latent_series)

  if existing_params and all(k in existing_params for k in ["gamma", "noise_std", "absolute_thresholds"]):
    gamma = existing_params["gamma"]
    noise_std = existing_params["noise_std"]
    thresholds = existing_params["absolute_thresholds"]

    # underlying continuous signal
    continuous_signal = gamma * latent_series + rng.normal(0, noise_std, size=n_samples)
  else: 
    # RANDOM PARAMETERS
    gamma = float(rng.uniform(0.6, 1.5) * rng.choice([-1, 1]))
    noise_std = float(rng.uniform(0.2, 0.5))

    # underlying continuous signal
    continuous_signal = gamma * latent_series + rng.normal(0, noise_std, size=n_samples)

    # boundaries for the thresholds
    low_bound = float(np.quantile(continuous_signal, 0.05))
    high_bound = float(np.quantile(continuous_signal, 0.95))

    # random thresholds
    thresholds_raw = rng.uniform(low_bound, high_bound, size=n_classes - 1)
    thresholds_raw.sort()
    thresholds = [float(t) for t in thresholds_raw]

  data = np.digitize(continuous_signal, thresholds)

  params = {
    "gamma": gamma,
    "noise_std": noise_std,
    "absolute_thresholds": thresholds
  }
  return data, params