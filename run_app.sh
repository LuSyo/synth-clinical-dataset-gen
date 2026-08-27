#!/bin/bash

SEED=4

python3 src/main.py \
  --exp_name "framework_validation" \
  --run_name "optuna_p10_r100_lo_recall_lo_$SEED" \
  --mapping "specs/scm_10.json" \
  --optimiser "optuna" \
  --seed "$SEED"\
  --n_pop 40000 \
  --n_train 3000 \
  --n_test 1000 \
  --y_prevalence 0.1 \
  --s_prevalence 0.5 \
  --diff_y_prev_factor 1\
  --target_raw_auprc 0.5\
  --target_disp "recall"\
  --target_biased_recall_disp 0.08\
  --target_biased_ppv_disp 0\
  --disparity_tolerance 0.03\
  --max_retries 20
