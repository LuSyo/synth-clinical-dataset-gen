#!/bin/bash

python3 src/main.py \
  --exp_name "exp" \
  --run_name "6" \
  --mapping "specs/scm.json" \
  --n_samples 5000 \
  --s_prevalence 0.35 \
  --y_prevalence 0.20 \
  --max_retries 1 \
  --query "I need a dataset yielding a global AUPRC of at least 0.5, and a recall for group 0 of 0.2 less than group 1."