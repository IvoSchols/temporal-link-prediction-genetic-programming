import collections
import functools
import itertools
import multiprocessing
import os

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
import typer

import operator
import random
from deap import algorithms, base, creator, tools, gp

import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline

app = typer.Typer()

# region STRATEGIES


def _rescale(x: pd.Series, *, lower_bound: float = 0.2) -> pd.Series:
    """_rescale the provided array.

    Args:
      lower_bound: Instead of normalizing between 0 and 1, normalize between 
        lower_bound and 1.
    """
    lowest, highest = np.quantile(x, [0, 1])
    return lower_bound + (1-lower_bound)*(x-lowest)/(highest-lowest)


def quantile_25(array):
    return np.quantile(array, .25)

def quantile_75(array):
    return np.quantile(array, .75)

AGGREGATION_STRATEGIES = {
    'q0': np.min,
    'q25': quantile_25,
    'q50': np.median,
    'q75': quantile_75,
    'q100': np.max,
    'm0': np.sum,
    'm1': np.mean,
    'm2': np.var,
    'm3': scipy.stats.skew,
    'm4': scipy.stats.kurtosis
}

def difference(x):
    return x[1] - x[0]

NODEPAIR_STRATEGIES = {
    'sum': sum, 
    'diff': difference,
    'max': max, 
    'min': min
}
# endregion

# region 

##
# Custom genetic programming functions
##
#
def div(left, right):
    return np.divide(_rescale(left.astype(int)), _rescale(right.astype(int)))

def log(x: np.ndarray):
    return np.log(_rescale(x.astype(int)))

def exp(x: np.ndarray):
    return np.exp(_rescale(x.astype(int)))

def sqrt(x: np.ndarray):
    return np.sqrt(_rescale(x.astype(int)))

##  Setup allowed primitives in the tree
#
pset = gp.PrimitiveSet("MAIN", 1)  # '1' is the number of input arguments for the function
pset.addPrimitive(np.add, 2)
pset.addPrimitive(np.subtract, 2)
pset.addPrimitive(np.multiply, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(sqrt, 1)
pset.addPrimitive(exp, 1)
pset.addPrimitive(log, 1)


pset.addEphemeralConstant("rand100", functools.partial(random.randint, -100, 100))
pset.renameArguments(ARG0='x')

##  Setup the toolbox   
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # We aim to maximize the fitness
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()


toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


# endregion


# region FEATURES

def aa_time_aware(edgelist_mature, instances,
                  time_strategy, aggregation_strategy):
    df = edgelist_mature.assign(datetime=lambda x: _rescale(time_strategy(x['datetime'])))

    G = nx.from_pandas_edgelist(df, edge_attr=True, create_using=nx.MultiGraph)
    scores = list()
    for u, v in instances:
        score = [
            aggregation_strategy([e['datetime'] for e in G[u][z].values()]) *
            aggregation_strategy([e['datetime'] for e in G[v][z].values()]) /
            np.log(len(G[z]))
            for z in nx.common_neighbors(G, u, v)
        ]
        scores.append(sum(score))
    return scores


def na(edgelist_mature, instances, time_strategy, aggregation_strategy,
       nodepair_strategy):
    df = edgelist_mature.assign(datetime=lambda x: _rescale(time_strategy(x['datetime'])))

    G = nx.from_pandas_edgelist(df, edge_attr=True, create_using=nx.MultiGraph)
    scores = list()
    for u, v in instances:
        activity_u = aggregation_strategy(
            [e['datetime'] for nb in G[u] for e in G[u][nb].values()]
        )
        activity_v = aggregation_strategy(
            [e['datetime'] for nb in G[v] for e in G[v][nb].values()]
        )
        scores.append(nodepair_strategy([activity_u, activity_v]))
    return scores

def cn_time_aware(edgelist_mature, instances, time_strategy,
                  aggregation_strategy):
    df = edgelist_mature.assign(datetime=lambda x: _rescale(time_strategy(x['datetime'])))

    G = nx.from_pandas_edgelist(df, create_using=nx.MultiGraph, edge_attr=True)

    scores = list()
    for u, v in instances:
        score = [
            aggregation_strategy([e['datetime'] for e in G[u][z].values()]) +
            aggregation_strategy([e['datetime'] for e in G[v][z].values()])
            for z in nx.common_neighbors(G, u, v)
        ]
        scores.append(sum(score))
    return scores


def jc_time_aware(edgelist_mature, instances, time_strategy,
                  aggregation_strategy):
    df = edgelist_mature.assign(datetime=lambda x: _rescale(time_strategy(x['datetime'])))

    G = nx.from_pandas_edgelist(df, create_using=nx.MultiGraph, edge_attr=True)

    scores = list()
    for u, v in instances:
        # logger.debug('Get CN')
        cn = [
            aggregation_strategy([e['datetime'] for e in G[u][z].values()]) +
            aggregation_strategy([e['datetime'] for e in G[v][z].values()])
            for z in nx.common_neighbors(G, u, v)
        ]
        # logger.debug('Get all activity of nodes')
        all_u = [aggregation_strategy([e['datetime'] for e in G[u][a].values()])
                 for a in G[u]]
        all_v = [aggregation_strategy([e['datetime'] for e in G[v][b].values()])
                 for b in G[v]]
        all_activity = sum(all_u) + sum(all_v)
        # logger.debug('Get score')
        score = sum(cn) / all_activity if all_activity != 0 else 0
        scores.append(score)
    return scores


def pa_time_aware(edgelist_mature, instances, time_strategy,
                  aggregation_strategy):
    df = edgelist_mature.assign(datetime=lambda x: _rescale(time_strategy(x['datetime'])))


    G = nx.from_pandas_edgelist(df, create_using=nx.MultiGraph, edge_attr=True)

    scores = list()
    for u, v in instances:
        all_u = [aggregation_strategy([e['datetime'] for e in G[u][a].values()])
                 for a in G[u]]
        all_v = [aggregation_strategy([e['datetime'] for e in G[v][b].values()])
                 for b in G[v]]
        scores.append(sum(all_u) * sum(all_v))
    return scores
# endregion

def calculate_feature(function, edgelist_mature, instances,
                        **kwargs):
    scores = function(edgelist_mature, instances, **kwargs)
    return scores

def time_func_helper(compiled_func, x):
    return compiled_func(_rescale(x.astype(int)))


# Evaluate the fitness of an individual. This is the function that will be
def eval_auc(individual, edgelist_mature, instances, agg_strategies, time_aware_funcs, y):      
    # NA -> only interested in II-A so skip


    # Time aware functions
    compiled_time_func = toolbox.compile(expr=individual)
    # Wrap time_func in _rescale(x.astype(int))
    time_func = functools.partial(time_func_helper, compiled_time_func)
    
    # Store scores in array
    X = {}

    
    for agg_str, agg_func in agg_strategies.items():
        for func_str, func in time_aware_funcs:
            feature = calculate_feature(
                func, edgelist_mature=edgelist_mature, instances=instances,
                time_strategy=time_func, aggregation_strategy=agg_func
            )
            X[f'{func_str}_{agg_str}'] = feature

    X = pd.DataFrame(X).fillna(0)
    X_train, X_test, y_train, y_test = (sklearn.model_selection.train_test_split(X, y, random_state=42))


    pipe = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(),
        sklearn.linear_model.LogisticRegression(max_iter=10000, random_state=42))
    pipe.fit(X_train, y_train)   

    auc = sklearn.metrics.roc_auc_score(
        y_true=y_test, y_score=pipe.predict_proba(X_test)[:,1])

    return (auc,)


@app.command()
def single(network:int, path: str, n_jobs: int = -1, verbose=True):
    edgelist_file = os.path.join(path, 'edgelist.pkl')
    samples_file = os.path.join(path, 'samples.pkl')
    if (not os.path.isdir(path) or
        not os.path.isfile(edgelist_file) or
            not os.path.isfile(samples_file)):
        return

    edgelist_mature = (
        pd.read_pickle(edgelist_file)
        .query("(phase == 'mature') & (source != target)")
    )
    instances = np.array([i for i in pd.read_pickle(samples_file).index])

    # Setup GP
    time_aware_funcs = [('aa', aa_time_aware),
                    ('jc', jc_time_aware),
                    ('cn', cn_time_aware),
                    ('pa', pa_time_aware)]


    
    # Enable multiprocessing for individuals
    # Initialize the multiprocessing pool with 30 workers
    core_count = 30
    pool = multiprocessing.Pool(core_count)
    toolbox.register("map", pool.map)


    # Prepare data
    # Replace NaNs with 0 -> not sure if this is the best way to handle this
    edgelist_mature = edgelist_mature[['source', 'target', 'datetime']]
    y = pd.read_pickle(samples_file).astype(int).values

    toolbox.register("evaluate", eval_auc, edgelist_mature=edgelist_mature, instances=instances, agg_strategies=AGGREGATION_STRATEGIES, time_aware_funcs=time_aware_funcs, y=y)

    random.seed(42)
    population_size = 200
    generations = 40
    keep_fittest_n = 10

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(keep_fittest_n)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)



    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, stats=stats, halloffame=hof, verbose=True)

    # Close the multiprocessing pool
    pool.close()
    pool.join()

    # Print logbook
    for record in logbook:
        print(record)

    # Print the best individuals
    for individual in hof:
        print("Fitness: ", individual.fitness, ", Individual: ", individual)

    # Write the results to a file corresponding to the network.txt
    with open(os.path.join(path, f'network_{network}_gp_results.txt'), 'w') as f:
        f.write(f'Fitness: {individual.fitness}, Individual: {individual}')


@app.command()
def check():
    iterator = list(
        itertools.product(
            [network for network in np.arange(1, 31)
             if network not in [15, 17, 26, 27]],
            np.arange(-100, 101, 20)
        )
    )
    result = dict()
    for n, nswap_perc in iterator:
        dir = f'data/{n:02}/{nswap_perc:+04.0f}/features'
        if os.path.isdir(dir):
            result[(n, nswap_perc)] = len(list(os.scandir(dir)))
    print(result)

if __name__ == '__main__':
    app()
