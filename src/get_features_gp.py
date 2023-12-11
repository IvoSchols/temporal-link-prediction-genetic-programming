import collections
import functools
import itertools
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


# def _exp_time(x: pd.Series) -> pd.Series:
#     """Apply y=3*exp(x) and normalize it between (0,1)."""
#     return np.exp(3*x) / np.exp(3)


# def lin(x: pd.Series, lower_bound=.2):
#     return _rescale(_rescale(x.astype(int)), lower_bound=lower_bound)


# def exp(x: pd.Series, lower_bound=.2):
#     return _rescale(_exp_time(_rescale(x.astype(int))), lower_bound=lower_bound)


# def sqrt(x: pd.Series, lower_bound=.2):
#     return _rescale(np.sqrt(_rescale(x.astype(int))), lower_bound=lower_bound) #type: ignore

AGGREGATION_STRATEGIES = {
    'q0': np.min,
    'q25': lambda array: np.quantile(array, .25),
    'q50': np.median,
    'q75': lambda array: np.quantile(array, .75),
    'q100': np.max,
    'm0': np.sum,
    'm1': np.mean,
    'm2': np.var,
    'm3': scipy.stats.skew,
    'm4': scipy.stats.kurtosis
}

NODEPAIR_STRATEGIES = {
    'sum': sum, 
    'diff': lambda x: x[1]-x[0],
    'max': max, 
    'min': min
}
# endregion

# region FEATURES

def aa_time_aware(edgelist_mature, instances,
                  time_strategy, aggregation_strategy):
    df = edgelist_mature[['source', 'target', 'datetime']].assign(
        datetime=lambda x: time_strategy(x['datetime'])
    )

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
    df = edgelist_mature[['source', 'target', 'datetime']].assign(
        datetime=lambda x: time_strategy(x['datetime'])
    )

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
    df = edgelist_mature[['source', 'target', 'datetime']].assign(
        datetime=lambda x: time_strategy(x['datetime'])
    )

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
    df = edgelist_mature[['source', 'target', 'datetime']].assign(
        datetime=lambda x: time_strategy(x['datetime'])
    )

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
    df = edgelist_mature[['source', 'target', 'datetime']].assign(
        datetime=lambda x: time_strategy(x['datetime'])
    )


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


@app.command()
def single(path: str, n_jobs: int = -1, verbose=True):

    def calculate_feature(function, edgelist_mature, instances,
                          **kwargs):
        scores = function(edgelist_mature, instances, **kwargs)
        return scores

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
    ##
    # Custom genetic programming functions
    ##
    #
    
    ##  Setup allowed primitives in the tree
    #
    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    pset = gp.PrimitiveSet("MAIN", 1)  # '1' is the number of input arguments for the function
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)  # A custom division function to protect against division by zero
    # pset.addPrimitive(math.cos, 1)
    # pset.addPrimitive(math.sin, 1)
    pset.addEphemeralConstant("rand100", functools.partial(random.randint, -100, 100))
    pset.renameArguments(ARG0='x')

    ##  Setup the toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # We aim to maximize the fitness
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # Evaluate the fitness of an individual. This is the function that will be
    y = pd.read_pickle(samples_file).astype(int).values

    def eval_auc(individual):      
        # NA -> only interested in II-A so skip


        # Time aware functions
        compiled_time_func = toolbox.compile(expr=individual)
        # Wrap time_func in _rescale(x.astype(int))
        time_func = lambda x: compiled_time_func(_rescale(x.astype(int)))
        
        # Store scores in array
        X = {}

        for agg_str, agg_func in AGGREGATION_STRATEGIES.items():
            for func_str, func in time_aware_funcs:
                feature = calculate_feature(
                    func, edgelist_mature=edgelist_mature, instances=instances,
                    time_strategy=time_func, aggregation_strategy=agg_func
                )
                X[f'{func_str}_{agg_str}'] = feature

        # Replace NaNs with 0 -> not sure if this is the best way to handle this
        X = pd.DataFrame(X).fillna(0)

        X_train, X_test, y_train, y_test = (
        sklearn.model_selection.train_test_split(X, y, random_state=42))
   
        pipe = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.linear_model.LogisticRegression(max_iter=10000, n_jobs=n_jobs, 
                                                    random_state=42))
        pipe.fit(X_train, y_train)   
  
        auc = sklearn.metrics.roc_auc_score(
            y_true=y_test, y_score=pipe.predict_proba(X_test)[:,1])

        return (auc,)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", eval_auc)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

  

    random.seed(42)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats=stats, halloffame=hof, verbose=True)

    print('done')


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
