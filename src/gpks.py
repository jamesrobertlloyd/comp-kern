"""
Simple interface to Gaussian process kernel search

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          
Created Oct 2014
"""

import numpy as np
import os
import re
import warnings
from multiprocessing import Pool
# noinspection PyUnresolvedReferences
from numpy.random import normal
# noinspection PyUnresolvedReferences
from numpy import log

import gp_model as gpm
from gp_model import GPModel
import grammar
import utils.misc


def optimise_single_model(params):
    """Target of multiprocessing"""
    model, X, Y, kwargs = params
    return model.gpy_optimize(X=X, Y=Y, **kwargs)


def perform_kernel_search(X, Y, exp):
    """Search for the best kernel"""
    # Initialise random seeds - randomness may be used in e.g. data subsetting
    utils.misc.set_all_random_seeds(exp['random_seed'])

    # Create location, scale and minimum period parameters to pass around for parameter initialisations
    data_shape = dict()
    data_shape['x_mean'] = [np.mean(X[:, dim]) for dim in range(X.shape[1])]
    data_shape['y_mean'] = np.mean(Y)  # TODO - need to rethink this for non real valued data
    data_shape['x_sd'] = log([np.std(X[:, dim]) for dim in range(X.shape[1])])
    data_shape['y_sd'] = log(np.std(Y))  # TODO - need to rethink this for non real valued data
    data_shape['y_min'] = np.min(Y)
    data_shape['y_max'] = np.max(Y)
    data_shape['x_min'] = [np.min(X[:, dim]) for dim in range(X.shape[1])]
    data_shape['x_max'] = [np.max(X[:, dim]) for dim in range(X.shape[1])]

    # Initialise period at a multiple of the shortest / average distance between points, to prevent Nyquist problems.
    # This is ultimately a little hacky and is avoiding more fundamental decisions
    if exp['period_heuristic_type'] == 'none':
        data_shape['min_period'] = None
    if exp['period_heuristic_type'] == 'min':
        data_shape['min_period'] = log([exp['period_heuristic'] * utils.misc.min_abs_diff(X[:, i])
                                        for i in range(X.shape[1])])
    elif exp['period_heuristic_type'] == 'average':
        data_shape['min_period'] = log([exp['period_heuristic'] * np.ptp(X[:, i]) / X.shape[0]
                                        for i in range(X.shape[1])])
    elif exp['period_heuristic_type'] == 'both':
        data_shape['min_period'] = log([max(exp['period_heuristic'] * utils.misc.min_abs_diff(X[:, i]),
                                               exp['period_heuristic'] * np.ptp(X[:, i]) / X.shape[0])
                                        for i in range(X.shape[1])])
    else:
        warnings.warn('Unrecognised period heuristic type : using most conservative heuristic')
        data_shape['min_period'] = log([max(exp['period_heuristic'] * utils.misc.min_abs_diff(X[:, i]),
                                            exp['period_heuristic'] * np.ptp(X[:, i]) / X.shape[0])
                                        for i in range(X.shape[1])])

    data_shape['max_period'] = [log((1.0 / exp['max_period_heuristic']) *
                                    (data_shape['x_max'][i] - data_shape['x_min'][i]))
                                for i in range(X.shape[1])]

    # Initialise mean, kernel and likelihood
    m = eval(exp['mean'])
    k = eval(exp['kernel'])
    l = eval(exp['lik'])
    current_models = [gpm.GPModel(mean=m, kernel=k, likelihood=l, ndata=Y.size)]

    print('\n\nStarting search with this model:\n')
    print(current_models[0].pretty_print())
    print('')

    # Perform the initial expansion

    # current_models = grammar.expand_models(D=X.shape[1],
    #                                        models=current_models,
    #                                        base_kernels=exp['base_kernels'],
    #                                        rules=exp['search_operators'])

    # Convert to additive form if desired

    if exp['additive_form']:
        current_models = [model.additive_form() for model in current_models]
        current_models = gpm.remove_duplicates(current_models)

    # Setup lists etc to record search and current state
    
    all_results = []  # List of scored kernels
    results_sequence = []  # List of lists of results, indexed by level of expansion.
    nan_sequence = []  # List of list of nan scored results
    oob_sequence = []  # List of list of out of bounds results
    best_models = None
    best_score = np.Inf

    # Setup multiprocessing pool

    processing_pool = Pool(processes=exp['n_processes'], maxtasksperchild=exp['max_tasks_per_process'])

    try:
    
        # Perform search
        for depth in range(exp['max_depth']):

            # If debug reduce number of models for fast evaluation
            if exp['debug']:
                current_models = current_models[0:4]

            # Add random restarts to kernels
            current_models = gpm.add_random_restarts(current_models, exp['n_rand'], exp['sd'], data_shape=data_shape)

            # Print result of expansion if debugging
            if exp['debug']:
                print('\nRandomly restarted kernels\n')
                for model in current_models:
                    print(model.pretty_print())

            # Remove any redundancy introduced into kernel expressions
            current_models = [model.simplified() for model in current_models]
            # Print result of simplification
            if exp['debug']:
                print('\nSimplified kernels\n')
                for model in current_models:
                    print(model.pretty_print())

            # Remove duplicate kernels
            current_models = gpm.remove_duplicates(current_models)
            # Print result of duplicate removal
            if exp['debug']:
                print('\nDuplicate removed kernels\n')
                for model in current_models:
                    print(model.pretty_print())

            # Add jitter to parameter values (helps sticky optimisers)
            current_models = gpm.add_jitter(current_models, exp['jitter_sd'])
            # Print result of jitter
            if exp['debug']:
                print('\nJittered kernels\n')
                for model in current_models:
                    print model.pretty_print()

            # Add the previous best models - in case we just need to optimise more rather than changing structure
            if not best_models is None:
                for a_model in best_models:
                    # noinspection PyUnusedLocal
                    current_models = current_models + [a_model.copy()] +\
                                     gpm.add_jitter([a_model.copy() for dummy in range(exp['n_rand'])], exp['jitter_sd'])

            # Randomise the order of the model to distribute computational load evenly if running on cluster
            np.random.shuffle(current_models)

            # Print current models
            if exp['debug']:
                print('\nKernels to be evaluated\n')
                for model in current_models:
                    print(model.pretty_print())

            if exp['strategy'] == 'vanilla':
                subset_n = X.shape[0]  # No subset
            elif exp['strategy'] == 'subset':
                subset_n = min(exp['starting_subset'], X.shape[0])
            while subset_n <= X.shape[0]:
                # Subset data
                X_subset = X[:subset_n]
                Y_subset = Y[:subset_n]

                # Use multiprocessing pool to optimise models
                kwargs = dict(inference='exact',
                              messages=exp['verbose'],
                              max_iters=exp['iters'])
                new_results = processing_pool.map(optimise_single_model,
                                                  ((model, X_subset, Y_subset, kwargs)
                                                   for model in current_models))

                # Remove models that were optimised to be out of bounds (this is similar to a 0-1 prior)
                # TODO - put priors on hyperparameters
                new_results = [a_model for a_model in new_results if not a_model.out_of_bounds(data_shape)]
                oob_results = [a_model for a_model in new_results if a_model.out_of_bounds(data_shape)]
                oob_results = sorted(oob_results, key=lambda a_model: GPModel.score(a_model, exp['score']), reverse=True)
                oob_sequence.append(oob_results)

                # Some of the scores may have failed - remove nans to prevent sorting algorithms messing up
                (new_results, nan_results) = remove_nan_scored_models(new_results, exp['score'])
                nan_sequence.append(nan_results)
                assert(len(new_results) > 0) # FIXME - Need correct control flow if this happens

                # Sort the new results
                new_results = sorted(new_results, key=lambda a_model: GPModel.score(a_model, exp['score']),
                                     reverse=True)

                # Keep only the top models
                if exp['strategy'] == 'subset':
                    new_results = new_results[int(np.floor(len(new_results) * exp['subset_pruning'])):]

                # Current = new
                current_models = new_results

                # Double the subset size, or exit loop if finished
                if subset_n == X.shape[0]:
                    break
                else:
                    subset_n = min(subset_n * 2, X.shape[0])

            # Update user
            print('\nAll new results\n')
            for model in new_results:
                print('BIC=%0.1f' % model.bic,
                      # 'NLL=%0.1f' % model.nll,
                      # 'AIC=%0.1f' % model.aic,
                      # 'PL2=%0.3f' % model.pl2,
                      model.pretty_print())

            all_results = all_results + new_results
            all_results = sorted(all_results, key=lambda a_model: GPModel.score(a_model, exp['score']), reverse=True)

            results_sequence.append(all_results)

            # Extract the best k kernels from the new all_results
            best_results = sorted(new_results, key=lambda a_model: GPModel.score(a_model, exp['score']))[0:exp['k']]

            # Print best kernels if debugging
            if exp['debug']:
                print('\nBest models\n')
                for model in best_results:
                    print model.pretty_print()

            # Expand the best models
            current_models = grammar.expand_models(D=X.shape[1],
                                                   models=best_results,
                                                   base_kernels=exp['base_kernels'],
                                                   rules=exp['search_operators'])

            # Print expansion if debugging
            if exp['debug']:
                print('\nExpanded models\n')
                for model in current_models:
                    print(model.pretty_print())

            # Convert to additive form if desired
            if exp['additive_form']:
                current_models = [model.additive_form() for model in current_models]
                current_models = gpm.remove_duplicates(current_models)

                # Print expansion
                if exp['debug']:
                    print('\Converted into additive\n')
                    for model in current_models:
                        print(model.pretty_print())

            # Reduce number of kernels when in debug mode
            if exp['debug']:
                current_models = current_models[0:4]

            # Have we hit a stopping criterion?
            if 'no_improvement' in exp['stopping_criteria']:
                new_best_score = min(GPModel.score(a_model, exp['score']) for a_model in new_results)
                if new_best_score < best_score - exp['improvement_tolerance']:
                    best_score = new_best_score
                else:
                    # Insufficient improvement
                    print 'Insufficient improvement to score - stopping search'
                    break

    finally:
        processing_pool.close()
        processing_pool.join()

    return all_results


def remove_nan_scored_models(models, score):
    not_nan = [m for m in models if not np.isnan(gpm.GPModel.score(m, criterion=score))]
    eq_nan = [m for m in models if np.isnan(gpm.GPModel.score(m, criterion=score))]
    return not_nan, eq_nan


def exp_param_defaults(exp_params):
    """Sets all missing parameters to their default values"""
    defaults = dict(data_dir=os.path.join('..', 'data', 'debug'),       # Where to find the datasets.
                    results_dir=os.path.join('..', 'results', 'debug'), # Where to write the results.
                    description='Default description',
                    max_depth=4,                  # How deep to run the search.
                    random_order=False,           # Randomize the order of the datasets?
                    k=1,                          # Keep the k best kernels at every iteration. 1 => greedy search.
                    debug=False,                  # Makes search simpler in various ways to keep compute cost down
                    n_rand=4,                     # Number of random restarts.
                    sd=2,                         # Standard deviation of random restarts.
                    jitter_sd=0.1,                # Standard deviation of jitter.
                    max_jobs=500,                 # Maximum number of jobs to run at once on cluster.
                    verbose=True,                 # Talkative?
                    skip_complete=True,           # Whether to re-run already completed experiments.
                    iters=50,                     # How long to optimize hyperparameters for.
                    base_kernels='SE,Noise',      # Base kernels of language
                    additive_form=True,           # Restrict kernels to be in an additive form?
                    mean='gpm.MeanZero()',         # Starting mean - zero
                    kernel='gpm.NoiseKernel()',    # Starting kernel - noise
                    lik='gpm.LikGauss(sf=-np.Inf)',# Starting likelihood - delta likelihood
                    verbose_results=False,        # Whether or not to record all kernels tested
                    random_seed=42,               # Random seed
                    period_heuristic=10,          # The minimum number of data points per period (roughly)
                    max_period_heuristic=5,       # Min number of periods that must be observed to declare periodicity
                    subset=False,                 # Optimise on a subset of the data?
                    subset_size=250,              # Size of data subset
                    full_iters=0,                 # Number of iters to perform on full data after subset optimisation
                    bundle_size=1,                # Number of kernel evaluations per job sent to cluster
                    score='BIC',                  # Search criterion
                    period_heuristic_type='both', # Related to minimum distance between data or something else
                    stopping_criteria=['no_improvement'], # Other reasons to stop the search
                    improvement_tolerance=0.1, # Minimum improvement for no_improvement stopping criterion
                    n_processes=None,             # Number of processes in multiprocessing.pool - None means max
                    max_tasks_per_process=1,      # This is set to one (or a small #) whilst there is a GPy memory leak
                    strategy='vanilla',           # Vanilla means regular old kernel search
                    starting_subset=100,          # How many data points do we start scoring on is using subset strat
                    subset_pruning=0.5,           # What proportion of models to discard when using subset strat
                    search_operators=[('A', ('+', 'A', 'B'), {'A': 'kernel', 'B': 'base'}),
                                      ('A', ('*', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}),
                                      #('A', ('*-const', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}),
                                      ('A', 'B', {'A': 'kernel', 'B': 'base'}),
                                      #('A', ('CP', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                                      #('A', ('CW', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                                      #('A', ('B', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                                      #('A', ('BL', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                                      ('A', ('None',), {'A': 'kernel'})]
                    )
    # Iterate through default key-value pairs, setting all unset keys
    for key, value in defaults.iteritems():
        if not key in exp_params:
            exp_params[key] = value
    return exp_params


def exp_params_to_str(exp_params):
    result = "Running experiment:\n"
    for key, value in exp_params.iteritems():
        result += "%s = %s,\n" % (key, value)
    return result
   

def run_debug():
    """Run a quick experiment"""
    # Create some data
    X = normal(size=(100, 3))
    Y = np.sin(2 * np.pi * X[:, [0]]) + np.sin(2 * np.pi * X[:, [1]])
    Y += 0.1 * normal(size=Y.shape)
    # Set up kernel search parameters
    exp = dict()
    exp['verbose'] = False
    exp = exp_param_defaults(exp)
    # Run kernel search
    all_results = perform_kernel_search(X, Y, exp)
    # Results
    print('\n\nBest kernel found:')
    for model in all_results[-1:]:
        print('BIC=%0.1f' % model.bic, model.pretty_print())

if __name__ == "__main__":
    run_debug()
