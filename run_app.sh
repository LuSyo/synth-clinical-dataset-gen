#!/bin/bash

python3 src/main.py \
  --exp_name "06-14_sampled" \
  --run_name "1" \
  --mapping "specs/scm_with_bias.json" \
  --n_pop 20000 \
  --n_train 3000 \
  --n_test 2000 \
  --s_prevalence 0.35 \
  --y_prevalence 0.20 \
  --max_retries 1 \
  --query "I need a dataset yielding a global AUPRC of roughly at least 0.48, and a recall for group 0 of at least 0.1 BELOW group 1."