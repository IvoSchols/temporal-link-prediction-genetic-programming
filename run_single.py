import numpy as np
import os
import src.get_performance as performance

def get_performance(network, nswap_perc=None, clf='LogisticRegression', feature_set='II-A', random_state=42, n_jobs=-1):
    performance.single(network, nswap_perc, clf, feature_set, random_state, n_jobs)

if __name__ == '__main__':
    # Set your desired parameters
    network = 1
    nswap_perc = 0 # What is this?
    clf = 'LogisticRegression'
    feature_set = 'II-A'
    random_state = 42
    n_jobs = -1


    # Add get_performance
    get_performance(network, nswap_perc, clf, feature_set, random_state, n_jobs)
