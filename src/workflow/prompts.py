class PipelinePrompts:

  RAW_DATA_VALIDATION_PROMPT = (
    "Task: Reparametrise baseline feature distributions to satisfy predictive capacity.\n\n"
    "Objective (Trial {current_trial}):\n"
    "- Target Global AUPRC: >= {target_raw_auprc}\n"
    "- Current Global AUPRC: {current_auprc}\n"
    "- Scope: Optimize Global AUPRC only; completely ignore subgroup disparities.\n\n"
    "{formulas_context}\n\n"
    "Parameter History:\n"
    "{feature_map}\n\n"
    "Performance History:\n"
    "{probe_results}\n\n"
    "Tuning Strategy:\n"
    "- Only adjust feature parameters: 'gamma', 'beta', 'noise_std', 'absolute_thresholds'.\n"
    "- Trials 0-1: Shift parameters with large deltas (0.4 to 1.5).\n"
    "- Trials 2+: Micro-adjust parameters with small deltas (0.05 to 0.1).\n"
    "Emit parameter overrides for the next trial."
  )

  BIASED_DATA_VALIDATION_PROMPT = (
    "Task: Tune sociological bias parameters to achieve target disparities.\n\n"
    "Objectives (Trial {current_trial}):\n"
    "- Target Recall Disparity: {recall_disp_target_str} (Current: {current_recall_disp})\n"
    "- Target PPV Disparity: {ppv_disp_target_str} (Current: {current_ppv_disp})\n"
    "- Scope: Optimize disparities only; completely ignore Global AUPRC.\n\n"
    "{bias_context}\n\n"
    "Bias Parameter History (soc pathway):\n"
    "{feature_map}\n\n"
    "Performance History:\n"
    "{probe_results}\n\n"
    "Tuning Strategy:\n"
    "- Modify bias parameters for 'soc' features.\n"
    "- Early Trials: Increase bias magnitude aggressively (e.g., lower alpha, increase p_suppress / p_down).\n"
    "- Late Trials: Micro-adjust around the historical best trial configuration.\n"
    "Emit updated bias parameters overrides for the next trial."
  )