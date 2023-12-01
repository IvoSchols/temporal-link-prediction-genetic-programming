import numpy as np
import os
import src


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

def rewire(network, nswap_perc, max_tries=1_000_000_000, seed=None, verbose=False):
    src.rewire.single(network, nswap_perc, max_tries, seed, verbose)

def get_samples(network, nswap_perc, cutoff=2, sample_size=10_000, seed=42, verbose=True):
    src.get_samples.single(network, nswap_perc, cutoff, sample_size, seed, verbose)

def get_features_gp(path, n_jobs=-1, verbose=True):
    src.get_features_gp.single(path, n_jobs, verbose)

def get_performance_gp(network, nswap_perc=None, clf='LogisticRegression', feature_set='II-A', random_state=42, n_jobs=-1):
    src.get_performance_gp.single(network, nswap_perc, clf, feature_set, random_state, n_jobs)

if __name__ == '__main__':
    # Set your desired parameters
    network = 1

    nswap_perc = 0 # What is this?
    # Call get_edgelist
    edgelist_path = f'data/{network:02}/edgelist.pkl'
    get_edgelist(network, edgelist_path)


    # Call rewire
    shuffle = True
    seed = 42
    verbose = False
    rewire(network, nswap_perc, seed=seed, verbose=verbose)

    # Call get_samples
    sample_size = 10_000
    get_samples(network, nswap_perc, sample_size=sample_size)

    # Get features gp (i.e. dont apply time strategy)
    path = f'data/{network:02}/{nswap_perc:+04.0f}/'
    n_jobs = -1
    verbose = True
    get_features_gp(f'data/{network:02}/samples.pkl', n_jobs, verbose)

    # Generate time strategies, using gp, and evaluate their performance. Keep the fittest
    # TODO: replace while loop
    while True:
            

        # Get performance
        directory = f'data/{network:02}/{nswap_perc:+04.0f}'
        os.makedirs(directory, exist_ok=True)
        filepath_out = os.path.join(directory, 
                                    'properties', 
                                    f'{feature_set}_{clf}.float')
        if os.path.isfile(filepath_out):
            return
        auc = predict(directory, feature_set, clf, random_state, n_jobs)
        if auc is not None:
            with open(filepath_out, 'w') as file:
                file.write(str(auc))
      
        # Call get_performance
        clf = 'LogisticRegression'
        feature_set = 'II-A'
        random_state = 42
        n_jobs = -1
        get_performance(network, nswap_perc, clf, feature_set, random_state, n_jobs)