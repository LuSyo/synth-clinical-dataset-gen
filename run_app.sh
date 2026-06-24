#!/bin/bash

python3 src/main.py \
  --exp_name "mres_toy_dataset" \
  --run_name "8" \
  --mapping "specs/mres_scm_1.json" \
  --n_pop 40000 \
  --n_train 3000 \
  --n_test 1000 \
  --s_prevalence 0.5 \
  --y_prevalence 0.10 \
  --diff_y_prev_factor 2\
  --target_raw_auprc 0.4\
  --target_disp "recall"\
  --target_biased_recall_disp 0.15\
  --target_biased_ppv_disp 0\
  --disparity_tolerance 0.05\
  --max_retries 5 