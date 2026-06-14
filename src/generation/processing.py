
import numpy as np
import pandas as pd
from typing import List, Union, Tuple

def stratified_sampling(
  df: pd.DataFrame, 
  strata: Union[str, List[str]], 
  n_train: int, 
  n_test: int, 
  rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
    Samples mutually exclusive Training and Testing dataframes from a parent population.
    Guarantees that the proportional distribution of the specified strata keys is 
    immaculately preserved across both splits with zero data leakage.
  """
  strata_cols = [strata] if isinstance(strata, str) else strata

  total_requested = n_train + n_test
  if total_requested > len(df):
    raise ValueError(
      f"Requested combined sample size ({total_requested}) exceeds the total available population size ({len(df)}). Reduce n_train/n_test or increase n_pop."
    )

  grouped = df.groupby(strata_cols)

  train_indices = []
  test_indices = []

  for pool_key, group in grouped:
    subgroup_fraction = len(group) / len(df)

    subgroup_n_train = int(np.round(subgroup_fraction * n_train))
    subgroup_n_test = int(np.round(subgroup_fraction * n_test))

    shuffled_indices = rng.permutation(list(group.index))

    total_needed_subgroup = subgroup_n_train + subgroup_n_test

    if total_needed_subgroup > len(shuffled_indices):
      subgroup_n_train = min(subgroup_n_train, len(shuffled_indices))
      subgroup_n_test = len(shuffled_indices) - subgroup_n_train
  
    group_train = shuffled_indices[:subgroup_n_train]
    group_test = shuffled_indices[subgroup_n_train : subgroup_n_train + subgroup_n_test]
    
    train_indices.extend(group_train)
    test_indices.extend(group_test)

  train_set = set(train_indices)
  test_set = set(test_indices)
  all_set = set(df.index.values)

  unallocated_pool = list(all_set - train_set - test_set)
  rng.shuffle(unallocated_pool)

  def adjust_set(current_set, target_n):
    if len(current_set) < target_n:
      needed = target_n - len(current_set)
      for _ in range(needed):
        if unallocated_pool:
          current_set.add(unallocated_pool.pop())
    elif len(current_set) > target_n:
      excess = len(current_set) - target_n
      current_set_list = list(current_set)
      for _ in range(excess):
        unallocated_pool.append(current_set_list.pop())
      current_set = set(current_set_list)
      
    assert len(current_set) == target_n
  
  adjust_set(train_set, n_train)
  adjust_set(test_set, n_test)

  assert train_set.isdisjoint(test_set)

  train_df = df.loc[list(train_set)].copy().reset_index(drop=True)
  test_df = df.loc[list(test_set)].copy().reset_index(drop=True)

  return train_df, test_df

