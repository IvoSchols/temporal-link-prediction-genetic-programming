import collections
import collections.abc
import os

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .core import get_edgelist_and_instances, Experiment
from ..helpers import file_exists, print_status

def single_source_number_paths_length_2(graph: nx.Graph, source):
  result = collections.Counter()
  for nb in graph[source]: 
    for nnb in graph[nb]: 
      result[nnb] += 1
  return result

def sp(path: str, verbose: bool = False) -> None:
  """Determine the number of shortest paths available for each nodepair in 
  instances.

  Args:
    path: The path should contain edgelist_mature.pkl and 
      instances_sampled.npy. Result is stored at path/sp.pkl.
      verbose (bool, optional): [description]. Defaults to False.
  """
  if verbose: print_status('Start sp(...)')
  
  feature_path = os.path.join(path, 'features')
  file = os.path.join(feature_path, 'sp.pkl')
  if file_exists(file, verbose=verbose): return 
  
  os.makedirs(feature_path, exist_ok=True)

  # Read in
  edgelist, instances = get_edgelist_and_instances(
    path, check_for_datetime=False, verbose=verbose)

  G = nx.from_pandas_edgelist(edgelist)

  # Slow method, but providing also shortest paths at greater distance:
#   [len(list(nx.all_shortest_paths(graph, *sample))) 
#    for sample in tqdm(instances, disable=not verbose, leave=False)]
  
  # Calculation
  paths_of_length_2 = {
    node: single_source_number_paths_length_2(G, node) 
    for node 
    in tqdm(instances[:,0], disable=not verbose, unit='node', leave=False)
  }
  
  scores = [paths_of_length_2[u][v] for u, v in instances]
  result = {Experiment('sp', time_aware=False): scores}

  # Store result
  if verbose: print_status('Store result')
  joblib.dump(result, file)