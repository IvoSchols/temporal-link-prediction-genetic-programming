import numpy as np
import os
import src


def get_edgelist(network, edgelist_path, split_fraction=2/3, t_min=None, t_split=None, t_max=None):
    src.get_edgelist.single(network, edgelist_path, split_fraction, t_min, t_split, t_max)

def rewire(network, nswap_perc, max_tries=1_000_000_000, seed=None, verbose=False):
    src.rewire.single(network, nswap_perc, max_tries, seed, verbose)

def get_samples(network, nswap_perc, cutoff=2, sample_size=10_000, seed=42, verbose=True):
    src.get_samples.single(network, nswap_perc, cutoff, sample_size, seed, verbose)

def get_performance(network, nswap_perc=None, clf='LogisticRegression', feature_set='II-A', random_state=42, n_jobs=-1):
    src.get_performance.single(network, nswap_perc, clf, feature_set, random_state, n_jobs)

if __name__ == '__main__':
    # Set your desired parameters
    network = 1


    # Call get_edgelist
    edgelist_path = f'data/{network:02}/edgelist.pkl'
    # Call rewire
    shuffle = True
    seed = 42
    verbose = False
    # Call get_samples
    sample_size = 10_000
    # Call get_performance
    nswap_perc = 0 # What is this?
    clf = 'LogisticRegression'
    feature_set = 'II-A'
    random_state = 42
    n_jobs = -1




    # Call functions
    get_edgelist(network, edgelist_path)
    rewire(network, nswap_perc, seed=seed, verbose=verbose)
    get_samples(network, nswap_perc, sample_size=sample_size)
    get_performance(network, nswap_perc, clf, feature_set, random_state, n_jobs)
