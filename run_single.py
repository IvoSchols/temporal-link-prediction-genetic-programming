import os
import numpy as np
import pandas as pd
import src
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline

##
# Helper function copied from get_features
##
def _rescale(x: pd.Series, *, lower_bound: float = 0.2) -> pd.Series:
    """_rescale the provided array.

    Args:
      lower_bound: Instead of normalizing between 0 and 1, normalize between 
        lower_bound and 1.
    """
    lowest, highest = np.quantile(x, [0, 1])
    return lower_bound + (1-lower_bound)*(x-lowest)/(highest-lowest)


def get_edgelist(network, edgelist_path, split_fraction=2/3, t_min=None, t_split=None, t_max=None):
    src.get_edgelist.single(network, edgelist_path, split_fraction, t_min, t_split, t_max)

def rewire(network, nswap_perc, max_tries=1_000_000_000, seed=42, verbose=False):
    src.rewire.single(network, nswap_perc, max_tries, seed, verbose)

def get_samples(network, nswap_perc, cutoff=2, sample_size=10_000, seed=42, verbose=True):
    src.get_samples.single(network, nswap_perc, cutoff, sample_size, seed, verbose)

def get_features(path, n_jobs=-1, verbose=True):
    # src.get_features.single(path, n_jobs, verbose)
    src.get_features_gp.single(path, n_jobs, verbose)

def get_performance(network, nswap_perc=None, clf='LogisticRegression', feature_set='II-A', random_state=42, n_jobs=-1):
    return src.get_performance.single(network, nswap_perc, clf, feature_set, random_state, n_jobs)

def get_performance(X,y):
    return src.get_performance.logistic_regression_auc(X,y)

if __name__ == '__main__':
    # Set your desired parameters
    network = 21

    nswap_perc = 0 # What is this?
    # Call get_edgelist
    edgelist_path = f'data/{network:02}/+000/edgelist.pkl'
    
    print('Getting edgelist')
    get_edgelist(network, edgelist_path, t_min=pd.Timestamp(2001, 1, 10) if network == 16 else None)


    # Call rewire
    print('Rewiring')
    rewire(network, nswap_perc)

    # Call get_samples
    print('Getting samples')
    get_samples(network, nswap_perc)

    # Get features gp (i.e. dont apply time strategy)
    path = f'data/{network:02}/{nswap_perc:+04.0f}/'
    verbose = True

    print('Getting features')
    get_features(f'data/{network:02}/+000/')

    # # Generate time strategies, using gp, and evaluate their performance. Keep the fittest
    # clf = 'LogisticRegression'
    # feature_set = 'II-A'
    # check = lambda x: not x.startswith('na') # DEPENDS ON FEATURE SET
    # random_state = 42

    # # Read in features of network
    # print('Reading in features')
    # directory = f'data/{network:02}/{nswap_perc:+04.0f}'
    # assert os.path.isdir(directory), f'missing {directory=}'
    # feature_dir = os.path.join(directory, 'features')
    # if not os.path.isdir(feature_dir):
    #    exit('No features found. Please run get_features first.') 
    # samples_filepath = os.path.join(directory, 'samples.pkl')
    # assert os.path.isfile(samples_filepath), f'missing {samples_filepath=}'
  
    # X = pd.DataFrame({
    #     f.name: np.load(f.path) 
    #     for f in os.scandir(feature_dir) if check(f.name)
    # })
    # y = pd.read_pickle(samples_filepath).astype(int).values

    print('Evaluating time strategies')

    