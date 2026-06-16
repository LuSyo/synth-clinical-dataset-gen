import pandas as pd
import numpy as np
import copy
from typing import Optional, Tuple

BIAS_FORMULAS_CONTEXT = """
### Sociological Bias Mechanics
  * Applied exclusively to specified features in the 'soc' pathway for the marginalized group (where S == 0).
  * True uncorrupted data remains in column 'feature_name'. The biased data is saved to column 'obs_feature_name' and passed to the downstream classifier instead of the original feature.
  * The four selectable bias execution types are defined as follows:

    1. type: "measurement_error" (Continuous features / additive bias)
      - Corrupts target group entries via a systematic mean shift and heteroscedastic noise.
      - Equation: X_obs = X_true + mu_bias + Normal(0, noise_std)
      - Parameters: mu_bias (directional shift), noise_std (noise standard deviation).

    2. type: "access_barrier" (Continuous features / acuity-based attenuation)
      - Attenuates mild/stable measurements for patients with limited clinical access.
      - Equation: If X_true < tau (empirical median): X_obs = X_true * alpha + Normal(0, noise_std)
      - Parameters: alpha (multiplicative attenuation factor where closer to 0 is more severe suppression), noise_std (noise standard deviation).

    3. type: "referral_bias" (Binary features / asymmetric entry suppression)
      - Systematically suppresses positive clinical recommendations or track entries for qualified patients.
      - Equation: If X_true == 1: P(X_obs = 1) = 1.0 - p_suppress. If X_true == 0: X_obs = 0.
      - Parameters: p_suppress (probability of a true 1 being overridden to an observed 0).

    4. type: "under_classification" (Categorical features / severity minimisation)
      - Down-stages recorded clinical severity levels while protecting baseline class boundaries.
      - Equation: If X_true > min_category_value: X_obs = X_true - 1 with stochastic probability (p_down * 0.95).
      - Parameters: p_down (probability of dropping down exactly 1 severity tier).
"""


def apply_feature_bias(
  df: pd.DataFrame,
  feature_map: dict,
  trial_index: int,
  rng: np.random.Generator
) -> Tuple[pd.DataFrame, dict]:
  """
  Applies a post-generation systematic bias to a feature array based on the 
  sensitive attribute S, dynamically initialising or preserving trial parameters.
  """
  df_copy = df.copy()
  feature_map_copy = copy.deepcopy(feature_map)
  trial_key = f"parameters_trial_{trial_index}"

  target_mask = (df_copy["S"] == 0)

  if "soc" in feature_map_copy:
    for feature in feature_map_copy["soc"]:
      name = feature["name"]
      bias_config = feature.get("bias")

      if bias_config:
        bias_type = bias_config["type"]
        true_data = df_copy[name].values

        existing_trial_params = feature.get(trial_key) or {}
        bias_params = existing_trial_params.get("bias_params", {})

        # -------------------------------------------------------------
        # 1. Measurement error (additive bias)
        # -------------------------------------------------------------
        if bias_type == "measurement_error":
          if not bias_params:
            bias_params = {
              "mu_bias": float(rng.uniform(-2.0, -0.5) * rng.choice([-1, 1])),
              "noise_std": float(rng.uniform(0.5, 1.5))
            }
          
          observed_data = apply_measurement_error(
            feature=true_data, 
            target_mask=target_mask, 
            mu_bias=bias_params['mu_bias'],
            noise_std=bias_params['noise_std'],
            rng=rng
          )

        # -------------------------------------------------------------
        # 2. Access barrier bias (acuity-based attenuation)
        # -------------------------------------------------------------
        elif bias_type == "access_barrier":
          if not bias_params:
            bias_params = {
              "threshold_quantile": 0.5,
              "alpha": float(rng.uniform(0.2, 0.5)),
              "noise_std": float(rng.uniform(0.05, 0.15))
            }
          
          observed_data = apply_access_barrier(
            feature=true_data,
            target_mask=target_mask,
            threshold_quantile=bias_params['threshold_quantile'],
            alpha=bias_params['alpha'],
            noise_std=bias_params['noise_std'],
            rng=rng
          )

        # -------------------------------------------------------------
        # 3. Referral bias
        # -------------------------------------------------------------
        elif bias_type == "referral_bias":
          if not bias_params:
            bias_params = {
              "p_suppress": float(rng.uniform(0.3, 0.7))
            }
              
          observed_data = apply_referral_bias(
            feature=true_data,
            target_mask=target_mask,
            p_suppress=bias_params['p_suppress'],
            rng=rng
          )

        # -------------------------------------------------------------
        # 4. Post-Generation Categorical Under-Staging Bias (Severity Mitigation)
        # -------------------------------------------------------------
        elif bias_type == "under_classification":
          if not bias_params:
            bias_params = {
              "p_down": float(rng.uniform(0.4, 0.8))
            }
              
          observed_data = apply_under_classification(
            feature=true_data,
            target_mask=target_mask,
            p_down=bias_params['p_down'],
            rng=rng
          )

        else:
          raise ValueError(f"Unsupported bias model execution token: '{bias_type}'")

        obs_col_name = f"obs_{name}"
        df_copy[obs_col_name] = observed_data

        if trial_key not in feature:
          feature[trial_key] = {}
        feature[trial_key]["bias_params"] = bias_params
        feature["observed_feature"] = obs_col_name

  return df_copy, feature_map_copy

def apply_measurement_error(feature, target_mask, mu_bias, noise_std, rng):
  """
    Applies a systematic shift to the target group's continuous feature.
    What it simulates: A medical device, diagnostic test, or clinical assessment tool that is systematically less accurate or miscalibrated when applied to a marginalized group.

    Inputs:
      - feature (pd.Series): true feature
      - target_mask (Boolean pd.Series): selection mask for the group affected by the bias
      - mu_bias (float): mean shift applied to the target group's feature
      - noise_std (float): stochastic noise std

    Output:
      - obs_feature (pd.Series): observed feature, where bias has been applied
  """
  obs_feature = feature.copy()
  n_targets = int(target_mask.sum())

  if n_targets > 0:
    noise = rng.normal(loc=mu_bias, scale=noise_std, size=n_targets)
    obs_feature[target_mask] += noise

  return obs_feature


def apply_access_barrier(feature, target_mask, threshold_quantile, alpha, noise_std, rng):
  """
    Applies an threshold-based attenuation to the target group's feature.
    What it simulates: Systemic healthcare access barriers where marginalised patients only present to the clinic or have their measurements recorded when their condition becomes severe. Mild or sub-acute presentations go systematically under-recorded or attenuated.

    Inputs:
      - feature (pd.Series): true feature
      - target_mask (Boolean pd.Series): selection mask for the group affected by the bias
      - threshold_quantile (float, (0,1)): clinical acuity threshold quantile under which attenuation is applied
      - alpha (float, (0,1)): multiplicative attenuation/suppression factor
      - noise_std (float): stochastic noise std 

    Output:
      - obs_feature (pd.Series): observed feature, where bias has been applied
  """
  obs_feature = feature.copy()

  # Calculate dynamic threshold based on the complete population's true distribution
  tau = float(np.quantile(obs_feature, threshold_quantile))

  low_acuity_target_mask = target_mask & (obs_feature < tau)
  n_low_acuity = int(low_acuity_target_mask.sum())

  if n_low_acuity > 0:
    # Apply attenuation with small non-deterministic variance
    attenuated_values = (obs_feature[low_acuity_target_mask] * alpha) + rng.normal(0, noise_std, size=n_low_acuity)
    obs_feature[low_acuity_target_mask] = attenuated_values

  return obs_feature


def apply_referral_bias(feature, target_mask, p_suppress, rng):
  """
    Applies a systematic suppression of the binary feature for the target group.
    What it simulates: Systemic or implicit practitioner bias in binary decision-making paths (e.g. choosing whether to refer a patient to a specialised cardiac unit, order an advanced scan, or document a highly subjective behavior).

    Inputs:
      - feature (pd.Series): true feature
      - target_mask (Boolean pd.Series): selection mask for the group affected by the bias
      - p_suppress (float, (0,1]): probability that a qualified minoritised patient gets their feature suppressed 

    Output:
      - obs_feature (pd.Series): observed feature, where bias has been applied
  """
  obs_feature = feature.copy()

  eligible_mask = target_mask & (obs_feature == 1)
  n_eligible = int(eligible_mask.sum())

  if n_eligible > 0:
    # Toss a biased coin for suppression rate
    suppression_trials = rng.binomial(n=1, p=1.0 - p_suppress, size=n_eligible)
    obs_feature[eligible_mask] = suppression_trials

  return obs_feature

def apply_under_classification(feature, target_mask, p_down, rng):
  """
    Applies a systematic down-grading of the ordinal feature for the target group.
    What it simulates: A systemic failure where clinical severity scales (e.g., triage notes, diagnostic staging, tool assessments) are recorded accurately for the majority group, but marginalized patients have their clinical severity systematically downgraded or under-coded due to systemic bias or stereotyping.

    Inputs:
      - feature (pd.Series): true feature
      - target_mask (Boolean pd.Series): selection mask for the group affected by the bias
      - p_down (float, (0,1]): probability that a qualified minoritised patient's severity level is down-graded.

    Output:
      - obs_feature (pd.Series): observed feature, where bias has been applied
  """
  obs_feature = feature.copy()

  min_ordinal = np.min(obs_feature)

  eligible_mask = target_mask & (obs_feature > min_ordinal)
  n_eligible = int(eligible_mask.sum())

  if n_eligible > 0:
    # Determine which records get downgraded stochastically
    downgrade_coins = rng.binomial(n=1, p=p_down, size=n_eligible)

    # Extra structural stochastic throttle to prevent a clean deterministic pattern
    throttle_coins = rng.binomial(n=1, p=0.95, size=n_eligible)
    
    final_downgrade_mask = (downgrade_coins * throttle_coins).astype(bool)
    
    eligible_indices = np.where(eligible_mask)[0]
    target_indices_to_drop = eligible_indices[final_downgrade_mask]
    
    obs_feature[target_indices_to_drop] -= 1

  return obs_feature