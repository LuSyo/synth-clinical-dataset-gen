#!/bin/bash

python3 src/main.py \
  --exp_name "exp" \
  --run_name "5" \
  --mapping "config/scm.json" \
  --n_samples 5000 \
  --s_prevalence 0.35 \
  --y_prevalence 0.20 \
  --query "hola"