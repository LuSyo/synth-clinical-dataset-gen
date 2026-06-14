class PipelinePrompts:

  DATASET_VALIDATION_PROMPT = (
    "You are an expert ML and Data Engineer assessing a synthetic dataset generation pipeline.\n"
    "Your objective is to verify if the generated dataset aligns with the user's approximate target predictive performance and group disparities as described in their original query.\n\n"
    "{formulas_context}\n"
    "{bias_context}\n\n"
    "# Multi-Trial Parameter Ledger: (Feature Map):\n"
    "{feature_map}\n\n"
    "# Latest Trial Dataset Summary (Table One):\n"
    "{table_one}\n\n"
    "# Chronological History of Downstream Classifier Performance & Disparities:\n"
    "{probe_results}\n\n"
    "# STRATEGIC OPTIMIZATION BOUNDARIES (CRITICAL):\n"
    "1. PATHWAY FOCUS CONSTRAINT: You are ONLY permitted to modify features residing in the 'soc' pathway category.\n"
    "   Do NOT emit parameter overrides for features in the 'bio' or 'ind' pathways under any circumstances.\n\n"
    "2. COOLING EXPLORATION SCHEDULE:\n"
    "   - Current Execution Status: Trial run {current_trial} "
    "   - If you are in an EARLY trial (Trial 0 or 1): Make BOLDER, aggressive moves! If a metric or disparity gap is far from target, swing the parameters significantly (e.g., scale gammas or adjust noise parameters by large deltas like 0.4 to 2) to rapidly explore the state space.\n"
    "   - If you are in a LATER trial (Trial 2+): Shift gears into fine-tuning exploitation mode. Make smaller, localized adjustments (e.g., tweaks of 0.05 to 0.1) to converge on the target numbers.\n\n"
    "Review the entire historical trajectory of metrics and parameters, calculate your current position in the exploration schedule. If the targets are NOT met, set is_acceptable to False and provide fine-tuned parameter adjustments strictly for 'soc' features in the 'adjusted_parameters' block. These adjustments will be applied to the next trial."
    )