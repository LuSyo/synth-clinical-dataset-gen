def format_raw_feature_history(feature_map: dict, max_trial: int) -> str:
  lines = []
  for pathway, feats in feature_map.items():
    for f in feats:
      name = f["name"]
      f_type = f.get("type", "unknown")
      lines.append(f"\n[{pathway}] {name} ({f_type}):")
      
      for t in range(max_trial + 1):
        key = f"parameters_trial_{t}"
        p = f.get(key)
        if not p:
          continue
        tag = " (Active)" if t == max_trial else ""
        
        if f_type == "continuous":
          lines.append(f"  - T{t}: gamma={p.get('gamma'):.2f}, beta={p.get('beta'):.2f}, noise_std={p.get('noise_std'):.2f}{tag}")
        elif f_type == "binary":
          lines.append(f"  - T{t}: gamma={p.get('gamma'):.2f}, beta={p.get('beta'):.2f}{tag}")
        elif f_type == "categorical":
          thresh = [round(x, 2) for x in p.get("absolute_thresholds", [])]
          lines.append(f"  - T{t}: gamma={p.get('gamma'):.2f}, thresholds={thresh}, noise_std={p.get('noise_std'):.2f}{tag}")
  return "\n".join(lines)

def format_bias_feature_history(feature_map: dict, max_trial: int) -> str:
  lines = []
  for f in feature_map.get("soc", []):
    name = f["name"]
    bias_info = f.get("bias", {})
    bias_type = bias_info.get("type", "none")
    lines.append(f"\n[soc] {name} ({bias_type}):")
    
    for t in range(max_trial + 1):
      key = f"parameters_trial_{t}"
      p = f.get(key, {}).get("bias_params")
      if not p:
        continue
      tag = " (Active)" if t == max_trial else ""
      params_str = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())
      lines.append(f"  - T{t}: {params_str}{tag}")
  return "\n".join(lines)