#!/bin/bash

python3 src/benchmark.py \
  --exp_name "framework_validation" \
  --mapping "specs/scm_10.json" \
  --benchmark_csv "results/framework_validation_0827/benchmark_datasets.csv" \
  --optimisers_list "llm" \
  --seeds_list 11 32 \
  --n_pop 40000 \
  --max_retries_optuna 20 \
  --max_retries_llm 10 \
  --n_train 4000 \
  --n_test 1500 \
  --n_boot 5