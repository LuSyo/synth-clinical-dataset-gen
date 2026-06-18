#!/bin/bash

python3 src/main.py \
  --exp_name "06-18_2" \
  --run_name "1" \
  --mapping "specs/scm.json" \
  --n_pop 20000 \
  --n_train 3000 \
  --n_test 2000 \
  --s_prevalence 0.5 \
  --y_prevalence 0.10 \
  --diff_y_prev_factor 2\
  --max_retries 0 \
  --query "no specific target expectations"