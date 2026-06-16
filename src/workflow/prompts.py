class PipelinePrompts:

  RAW_DATA_VALIDATION_PROMPT = (
    "You are an expert ML and Data Engineer assessing a synthetic dataset generation pipeline.\n"
    "Your objective is to verify if the raw dataset aligns with the user's target baseline predictive performance and initial subgroup disparities.\n\n"
    
    "{formulas_context}\n\n"
    
    "# Multi-Trial Parameter Ledger (Feature Map):\n"
    "{feature_map}\n\n"
    
    "# Latest Trial Dataset Summary (Table One):\n"
    "{table_one}\n\n"
    
    "# Chronological History of Downstream Classifier Performance on RAW Features:\n"
    "{probe_results}\n\n"
    
    "# STRATEGIC OPTIMIZATION BOUNDARIES (CRITICAL):\n"
    "1. PARAMETER SPACE CONSTRAINT: You are ONLY permitted to modify baseline generation parameters ('gamma', 'beta', 'noise_std') for features residing in the 'soc' pathway.\n"
    "   Do NOT modify or emit overrides for any 'bias_params' keys, 'bio' pathways, or 'ind' pathways.\n\n"
    
    "2. COOLING EXPLORATION SCHEDULE:\n"
    "   - Current Execution Status: Trial run {current_trial}\n"
    "   - Early Trials (0 or 1): Make BOLDER moves! Swing gammas or betas by larger intervals (e.g., 0.4 to 1.5) to find the right baseline space.\n"
    "   - Later Trials (2+): Shift into micro-fine-tuning mode. Adjust baseline parameters by small deltas (e.g., 0.05 to 0.1) to lock in target performance.\n\n"
    
    "Review the trajectory of raw metrics and parameters. If raw performance and disparity targets are NOT met, set is_acceptable to False and provide adjustments for 'soc' baseline keys ('gamma', 'beta', 'noise_std') in your response block. If baseline performance is acceptable, set is_acceptable to True."
  )

  BIASED_DATA_VALIDATION_PROMPT = (
    "You are an expert AI Fairwashing and Bias Simulation Engineer assessing a synthetic data degradation pipeline.\n"
    "Your objective is to tune post-generation sociological bias parameters to force the final observed dataset to match target operational disparities.\n\n"
    
    "{formulas_context}\n\n"
    "{bias_context}\n\n"
    
    "# Multi-Trial Parameter Ledger (Feature Map):\n"
    "{feature_map}\n\n"
    
    "# Chronological History of Downstream Classifier Performance:\n"
    "{probe_results}\n\n"
    
    "# STRATEGIC OPTIMIZATION BOUNDARIES (CRITICAL):\n"
    "1. BIAS PARAMETER CONSTRAINT: You are ONLY permitted to modify parameters inside the 'bias_params' dictionary block for features in the 'soc' pathway.\n"
    "   - For type 'measurement_error', you can adjust: 'mu_bias', 'noise_std'\n"
    "   - For type 'access_barrier', you can adjust: 'alpha', 'noise_std'\n"
    "   - For type 'referral_bias', you can adjust: 'p_suppress'\n"
    "   - For type 'under_classification', you can adjust: 'p_down'\n"
    "   CRITICAL: Do NOT modify baseline parameters ('gamma', 'beta') under any circumstances.\n\n"
    
    "2. SHARED BUDGET COOLING SCHEDULE:\n"
    "   - Global Execution Progress: Trial run {current_trial} (Budget shared across generation and bias phases)\n"
    "   - Early/Mid Trials: If the biased disparity gap is weak, aggressively strengthen bias values (e.g., lower alpha closer to 0, increase p_suppress or p_down closer to 1.0).\n"
    "   - Final Trials: Apply fine adjustments to stabilize metrics without breaking the classifier entirely.\n\n"
    
    "Compare the biased performance against the raw reference baseline. If target downstream disparity scales are not satisfied, set is_acceptable to False and emit fine-tuned parameter adjustments exclusively for keys inside the 'bias_params' sub-blocks of 'soc' features."
  )