import numpy as np
import pandas as pd
import json
import math
from typing import cast, Tuple, Dict, Any, Optional
from scipy.special import expit
from scipy.optimize import bisect
from scipy.stats import rankdata, norm

FEATURE_FORMULAS_CONTEXT = """
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
  - Baseline True Features (from Feature Map):
    * Continuous: 
      - All incoming parent latents are standard-normalized (Mean=0, SD=1) 
      - Normal: X = gamma * Normalized_Latent + beta + Normal(0, noise_std)
      - Lognormal: X = abs(gamma) * exp(sign(gamma) * Normalized_Latent + Normal(0, noise_std)) + beta
      - Note: The directional sign of gamma controls direct vs inverse latent correlation, while the absolute magnitude of gamma acts linear scale multiplier
    * Binary: P(X=1) = sigmoid(gamma * Latent + beta)
    * Categorical: Digitize an underlying continuous signal [gamma * Latent + Normal(0, noise_std)]
      - Note: The n-1 class boundaries are calculated between the 5th and 95th percentiles of the continuous signal.
  """

def generate_clinical_ground_truth(
  n_pop: int, 
  s_prevalence: float, 
  y_prevalence: float, 
  diff_y_prev_factor: float,
  rng: np.random.Generator,
  h_dim: int = 3,
  u_dep_dim: int = 2,
  u_indep_dim: int = 2,
) -> dict:
  """
  Generates the latent health state of the population, underlying cause of the outcome, as per the following SCM: 
  H -> U_dep
  H -> U_indep
  S -> U_dep
  U_dep -> Y
  U_indep -> Y

  Inputs:
    - n_pop (int): size of the population
    - s_prevalence (float, [0, 1]): prevalence of the majority group of the sensitive attribute S in the population
    - y_prevalence (float, [0, 1]): prevalence of the clinical outcome Y in the population
    - seed (int): random seed for reproducibility

  Outputs:
    - DataFrame with n_pop rows and the columns:
      - H: central health latent 
      - S: binary sensitive attribute
      - U_dep: Latent descending from H, directly influenced by S
      - U_indep: Latent descending from H, independent of S
  """
    
  # Binary Sensitive attribute S (Bernoulli/Binomial distribution)
  n_1 = math.floor(s_prevalence*n_pop)
  n_0 = n_pop - n_1
  S_1 = np.ones(shape=n_1, dtype=int)
  S_0 = np.zeros(shape=n_0, dtype=int)
  S = np.concat((S_1, S_0))
  rng.shuffle(S)
  
  # Central health latent H (Independent Gamma distribution matrix)
  h_shape, h_scale = 2.0, 1.5
  H = rng.gamma(shape=h_shape, scale=h_scale, size=(n_pop, h_dim))
  
  # Latent U_indep (Descends from H, independent of S)
  W_indep = rng.uniform(0.4, 0.8, size=((h_dim, u_indep_dim)))
  H_proj_indep = np.dot(H, W_indep)
  U_indep = H_proj_indep + rng.gamma(shape=2.1, scale=1, size=(n_pop, u_indep_dim))
  
  # Latent U_dep (Descends from H, directly influenced by S)
  W_dep = rng.uniform(0.4, 0.8, size=((h_dim, u_dep_dim)))
  H_proj_dep = np.dot(H, W_dep)

  min_multiplier = 1
  maj_multiplier = diff_y_prev_factor / 0.75 + min_multiplier
  multiplier = np.where(S == 0, min_multiplier, maj_multiplier).reshape(-1, 1)

  U_dep = H_proj_dep + multiplier * rng.gamma(shape=1.2, scale=1, size=(n_pop, u_dep_dim))
  
  # Outcome
  structural_signal = np.log(U_dep.mean(axis=1) + U_indep.mean(axis=1))
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
  
  return {
    "S": S,
    "Y": Y,
    "U_indep": U_indep,
    "U_dep": U_dep
  }

def generate_observed_features(
    ground_truth: dict,
    feature_map: dict,
    rng: np.random.Generator, 
    trial_index: int = 0
) -> Tuple[pd.DataFrame, dict]:
  """
    Generates the set of features from the pre-configured feature map and using the previously generated clinical ground truth
  """
  df = pd.DataFrame({
    "S": ground_truth["S"],
    "Y": ground_truth["Y"]
  })

  enriched_map = json.loads(json.dumps(feature_map))

  pathway_latent_map = {
    "bio": "U_dep",
    "soc": "U_indep",
    "ind": "U_indep"
  }

  latent_dim_counters = {
    "U_dep": 0,
    "U_indep": 0
  }

  for pathway_key, feature_list in enriched_map.items():
    if pathway_key not in pathway_latent_map:
      print(f"Warning: Pathway '{pathway_key}' is not mapped to an SCM latent. Skipping.")
      continue
        
    parent_latent_name = pathway_latent_map[pathway_key]
    
    for feature_spec in feature_list:
      name = feature_spec["name"]
      feat_type = feature_spec.get("type", "").lower()
      trial_key = f"parameters_trial_{trial_index}"
      existing_params = feature_spec.get(trial_key, None)

      # Map feature to latent dimension
      if existing_params and "latent_dim_idx" in existing_params:
        dim_idx = existing_params["latent_dim_idx"]
      else:
        dim_idx = latent_dim_counters[parent_latent_name]

      latent_dim_counters[parent_latent_name] += 1

      if dim_idx > ground_truth[parent_latent_name].shape[1] - 1:
        raise KeyError(f"Latent dimension assigned to feature {name} is outside of latent dimensionality.")

      latent_series = ground_truth[parent_latent_name][:, dim_idx]
      
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

      params['latent_dim_idx'] = dim_idx
      df[name] = data_array
      feature_spec[trial_key] = params
              
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
  n_pop = len(latent_series)

  # Standardise the latent
  ranks = rankdata(latent_series)
  percentiles = (ranks - 0.5) / n_pop
  perfect_normal_latent = norm.ppf(percentiles)

  latent_mean = float(latent_series.mean())
  latent_std = float(latent_series.std()) if latent_series.std() > 0 else 1.0
  norm_latent = (latent_series - latent_mean) / latent_std

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
    gamma_sign = np.sign(gamma) if gamma != 0 else 1.0
    gamma_magnitude = abs(gamma)
    unscaled_lognormal = np.exp(gamma_sign * perfect_normal_latent + rng.normal(0, noise_std, size=n_pop))
    data = (gamma_magnitude * unscaled_lognormal) + beta
    dist_type = "lognormal"
    
  else:  # Default to "normal"
    dist_type = "normal"
    data = gamma * perfect_normal_latent + beta + rng.normal(0, noise_std, size=n_pop)

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
  n_pop = len(latent_series)

  if existing_params and all(k in existing_params for k in ["gamma", "noise_std", "absolute_thresholds"]):
    gamma = existing_params["gamma"]
    noise_std = existing_params["noise_std"]
    thresholds = existing_params["absolute_thresholds"]

    # underlying continuous signal
    continuous_signal = gamma * latent_series + rng.normal(0, noise_std, size=n_pop)
  else: 
    # RANDOM PARAMETERS
    gamma = float(rng.uniform(0.6, 1.5) * rng.choice([-1, 1]))
    noise_std = float(rng.uniform(0.2, 0.5))

    # underlying continuous signal
    continuous_signal = gamma * latent_series + rng.normal(0, noise_std, size=n_pop)

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