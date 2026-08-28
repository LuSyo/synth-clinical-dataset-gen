import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, cast, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from tableone import TableOne


def plot_cont_feature(
  df: pd.DataFrame, 
  feature: str, 
  label: str, 
  hue_1: str, 
  hue_2: str,
  output_dir: str,
  ):

  plots_dir = os.path.join(output_dir, "plots")
  os.makedirs(plots_dir, exist_ok=True)

  if len(hue_2):
    layout = """
      AB
      CC
      """
    height = 9
  else:
    layout = "AC"
    height = 5

  fig, axes = plt.subplot_mosaic(layout, figsize=(16, height))
  sns.histplot(df, x=feature, hue=hue_1, bins=50, common_norm=False, multiple='dodge', kde=True, stat='probability', ax=axes['A'])
  if len(hue_2):
    sns.histplot(df, x=feature, hue=hue_2, bins=50, common_norm=False, multiple='dodge', kde=True, stat='probability', ax=axes['B'])
  sns.histplot(df, x=feature, bins=50, kde=True, stat='probability', ax=axes['C'])
  fig.suptitle(f'Probability distribution of {label}', fontsize=16)
  
  plt.savefig(os.path.join(plots_dir, f"hist_{feature}.png"))
  plt.close()

  return fig

def plot_cat_feature(
  df: pd.DataFrame, 
  feature: str, 
  label: str, 
  hue_1: str, 
  output_dir: str,
  hue_2: str = ""
  ):

  plots_dir = os.path.join(output_dir, "plots")
  os.makedirs(plots_dir, exist_ok=True)

  if len(hue_2):
    layout = """
      AB
      CC
      """
    height = 9
  else:
    layout = "AC"
    height = 4

  fig, axes = plt.subplot_mosaic(layout, figsize=(16, height))
  sns.histplot(df, x=df[feature].astype(int), hue=hue_1, common_norm=False, multiple='dodge', discrete=True, stat='probability', ax=axes['A'])
  if len(hue_2):
    sns.histplot(df, x=feature, hue=hue_2, common_norm=False, multiple='dodge', discrete=True, stat='probability', ax=axes['B'])
  sns.histplot(df, x=feature, discrete=True, stat='probability', ax=axes['C'])
  fig.suptitle(f'Probability distribution of {label}', fontsize=16)

  plt.savefig(os.path.join(plots_dir, f"hist_{feature}.png"))
  plt.close()

def create_table_one(
  df: pd.DataFrame, 
  groupby: str,
  output_dir: str,
  feature_map: dict
  ):
  continuous_features, categorical_features = get_feat_lists(df, feature_map)

  table1 = TableOne(df,
                  groupby=groupby,
                  continuous= continuous_features,
                  categorical=categorical_features + ["Y"],
                  missing=False,
                  sort=True
                  )

  formatted_table1 = table1.tabulate(tablefmt = "github")

  report_path = os.path.join(output_dir, "table_one.txt")
  with open(report_path, "w") as f:
    f.write(formatted_table1)

def get_feat_lists(df: pd.DataFrame, feature_map: dict) -> Tuple[List, List]:
  continuous_features = []
  categorical_features = []
  for pathway, feature_list in feature_map.items():
    for feature in feature_list:
      if feature['name'] in df.columns:
        if feature['type'].lower().strip() == "continuous":
          continuous_features.append(feature['name'])
        else:
          categorical_features.append(feature['name'])

      obs_name = feature.get('observed_feature')
      if obs_name and obs_name in df.columns:
        if feature['type'].lower().strip() == "continuous":
          continuous_features.append(obs_name)
        else:
          categorical_features.append(obs_name)

  return continuous_features, categorical_features

def plot_dataset(df: pd.DataFrame, feature_map: dict, output_dir: str):
  latents = ["H", "U_dep", "U_indep"]

  for L in latents:
    if L in df.columns:
      plot_cont_feature(
        df=df, feature=L, label=f"Latent {L}",
        hue_1="Y", hue_2="S",
        output_dir=output_dir
      )

  continuous_features, categorical_features = get_feat_lists(df, feature_map)

  for feature in continuous_features:
    plot_cont_feature(
      df=df, feature=feature, label=f"Feature {feature}",
      hue_1="Y", hue_2="S",
      output_dir=output_dir
    )

  for feature in categorical_features:
    plot_cat_feature(
      df=df, feature=feature, label=f"Feature {feature}",
      hue_1="Y", hue_2="S",
      output_dir=output_dir
    )
  
  # Outcome
  plot_cat_feature(
    df=df, feature="Y", label="Outcome",
    hue_1="S",
    output_dir=output_dir
  )
  # Sensitive Attribute
  plot_cat_feature(
    df=df, feature="S", label="Sensitive Attribute",
    hue_1="Y",
    output_dir=output_dir
  )
  
def run_downstream_probe(
  df: pd.DataFrame, 
  feature_map: dict, 
  n_train: int,
  n_test: int,
  current_phase: str,
  rng: np.random.Generator
  ) -> Tuple[str | None, float, float]:
  """
    Trains a RandomForest classifier probe across 4 bootstrap iterations.
    Evaluates predictive performance metrics globally and stratified across S subgroups,
    and saves the final aggregated summary report as a Markdown table.
  """
  print("---> Running Downstream Predictive Probe (4 Bootstrap Rounds)")

  S = df['S']
  subgroups = sorted(S.unique().astype(int))

  features = []
  for pathway in ['ind', 'bio', 'soc']:
    for f in feature_map.get(pathway, []):
      base_name = f['name']
      obs_name = f.get('observed_feature')
      
      if current_phase != "generation" and obs_name and obs_name in df.columns:
        features.append(obs_name)
      elif base_name in df.columns:
        features.append(base_name)

  X = df[features]
  y = df['Y']

  global_metrics = {"auprc": [], "recall": [], "precision": []}
  subgroup_metrics = {
    g: {"auprc": [], "recall": [], "precision": []} for g in subgroups
  }

  n_pop = len(df)
  indices = np.arange(n_pop)

  for boot_round in range(4):
    # Bootstrap training sample
    train_idx = rng.choice(indices, size=n_train, replace=True)
    # OOB test samples 
    oob_idx = np.setdiff1d(indices, train_idx)
    test_size = min(len(oob_idx), n_test)
    test_idx = rng.choice(oob_idx, size=test_size, replace=False)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    s_test = S.iloc[test_idx]

    # Fit and apply scaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=pd.Index(features), index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=pd.Index(features), index=X_test.index)
    
    # Fit probe exactly once per fold
    probe = RandomForestClassifier(
      n_estimators=300,
      min_samples_leaf=5,
      random_state=int(rng.integers(0, 100000))
    )
    probe.fit(X_train_scaled, y_train)

    raw_probs = cast(np.ndarray, probe.predict_proba(X_test_scaled))
    y_pred_prob = raw_probs[:, 1]

    global_threshold = y_train.sum()/len(y_train)
    y_pred = (y_pred_prob > global_threshold).astype(int)
    global_metrics["auprc"].append(average_precision_score(y_test, y_pred_prob))
    global_metrics["recall"].append(recall_score(y_test, y_pred))
    global_metrics["precision"].append(precision_score(y_test, y_pred))

    for g in subgroups:
      subgroup_mask = (s_test == g)

      if np.sum(subgroup_mask) == 0:
        continue

      y_test_sub = y_test[subgroup_mask]
      y_pred_prob_sub = y_pred_prob[subgroup_mask] 

      y_pred_sub = (y_pred_prob_sub > global_threshold).astype(int)

      if y_test_sub.sum() > 0:
        subgroup_metrics[g]["auprc"].append(average_precision_score(y_test_sub, y_pred_prob_sub))  
        subgroup_metrics[g]["recall"].append(recall_score(y_test_sub, y_pred_sub))
      else:
        subgroup_metrics[g]["auprc"].append(np.nan)
        subgroup_metrics[g]["recall"].append(np.nan)

      subgroup_metrics[g]["precision"].append(precision_score(y_test_sub, y_pred_sub))


  # --- AGREGATE RESULTS ---
  def fmt(vals):
    arr = np.array(vals)
    return f"{np.nanmean(arr):.3f} &plusmn; {np.nanstd(arr):.3f}"

  def disp(vals_1, vals_0):
    arr_1 = np.array(vals_1)
    arr_0 = np.array(vals_0)
    return round(float(np.nanmean(arr_1) - np.nanmean(arr_0)), 3)

  report_rows = []

  # Global results
  report_rows.append({
    "Cohort Slice": "**Global Population**",
    "AUPRC (mean &plusmn; sd)": fmt(global_metrics['auprc']),
    "Recall (mean &plusmn; sd)": fmt(global_metrics['recall']),
    "Precision (mean &plusmn; sd)": fmt(global_metrics['precision'])
  })

  # Stratified results
  for g in subgroups:
    group_label = f"Majority Group (S={g})" if g == 1 else f"Minority Group (S={g})"
    report_rows.append({
      "Cohort Slice": group_label,
      "AUPRC (mean &plusmn; sd)": fmt(subgroup_metrics[g]['auprc']),
      "Recall (mean &plusmn; sd)": fmt(subgroup_metrics[g]['recall']),
      "Precision (mean &plusmn; sd)": fmt(subgroup_metrics[g]['precision'])
    })

  recall_disparity = disp(subgroup_metrics[1]['recall'], subgroup_metrics[0]['recall'])
  precision_disparity = disp(subgroup_metrics[1]['precision'], subgroup_metrics[0]['precision'])
  mean_auprc = np.nanmean(global_metrics['auprc'])

  report_df = pd.DataFrame(report_rows)
  markdown_table = report_df.to_markdown(index=False)

  return markdown_table, mean_auprc, recall_disparity, precision_disparity

def evaluate_downstream_transferability(
  df: pd.DataFrame,
  n_train: int,
  n_test: int,
  n_boot: int,
  feature_map: dict,
  rng: np.random.Generator
):

  features = []
  for pathway in ['ind', 'bio', 'soc']:
    for f in feature_map.get(pathway, []):
      base_name = f['name']
      obs_name = f.get('observed_feature')
      
      if obs_name and obs_name in df.columns:
        features.append(obs_name)
      elif base_name in df.columns:
        features.append(base_name)

  n_pop = len(df)
  indices = np.arange(n_pop)

  model_names = ["lr", "rf", "mlp"]
  bootstrap_metrics = {
    name: {"auprc": [], "recall_disp": [], "ppv_disp": []} 
    for name in model_names
  }

  for _ in range(n_boot):
    train_df, test_df = train_test_split(
      df,
      test_size=n_test,
      train_size=n_train,
      stratify=df[['S', 'Y']],
      random_state=int(rng.integers(0, 100000))
    )

    X_train, y_train = train_df[features], train_df['Y']
    X_test, y_test = test_df[features], test_df['Y']
    s_test = test_df['S'].values

    models = {
      "lr": make_pipeline(
        StandardScaler(),
        LogisticRegression(
          max_iter=1000, 
          random_state=int(rng.integers(0, 100000)))
        ),
      "rf": RandomForestClassifier(
        n_estimators=100, 
        max_depth=6, 
        random_state=int(rng.integers(0, 100000))),
      "mlp": make_pipeline(
        StandardScaler(), 
        MLPClassifier(
          hidden_layer_sizes=(32, 16), 
          max_iter=500, 
          early_stopping=True, 
          random_state=int(rng.integers(0, 100000)))
      ),
    }

    global_threshold = y_train.sum()/len(y_train)

    for name, clf in models.items():
      clf.fit(X_train, y_train)
      y_prob = clf.predict_proba(X_test)[:, 1]
      y_pred = (y_prob > global_threshold).astype(int)
    
      auprc = average_precision_score(y_test, y_prob)

      rec_maj = recall_score(y_test[s_test == 1], y_pred[s_test == 1], zero_division=np.nan)
      rec_min = recall_score(y_test[s_test == 0], y_pred[s_test == 0], zero_division=np.nan)
      ppv_maj = precision_score(y_test[s_test == 1], y_pred[s_test == 1], zero_division=np.nan)
      ppv_min = precision_score(y_test[s_test == 0], y_pred[s_test == 0], zero_division=np.nan)

      bootstrap_metrics[name]["auprc"].append(auprc)
      bootstrap_metrics[name]["recall_disp"].append(rec_maj - rec_min)
      bootstrap_metrics[name]["ppv_disp"].append(ppv_maj - ppv_min)

  final_results = {}
  for name in model_names:
    for metric_key in ["auprc", "recall_disp", "ppv_disp"]:
      values = np.array(bootstrap_metrics[name][metric_key])
      final_results[f"downstream_{name}_{metric_key}_mean"] = float(np.nanmean(values))
      final_results[f"downstream_{name}_{metric_key}_std"] = float(np.nanstd(values))

  return final_results