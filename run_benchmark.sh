#!/bin/bash

python3 src/benchmark.py \
  --exp_name "framework_validation" \
  --mapping "specs/scm_10.json" \
  --benchmark_csv "benchmark_datasets.csv" \
  --optimisers_list "optuna" \
  --seeds_list 4 11 32 \
  --n_pop 40000 \
  --max_retries_optuna 20 \
  --max_retries_llm 10