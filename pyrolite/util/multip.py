import numpy as np

try:
    from pathos.multiprocessing import ProcessingPool as Pool
except ImportError:
    from multiprocessing import Pool

from .log import Handle

logger = Handle(__name__)

# Note : Using pathos multiprocessing which leverages dill over standard
# pickle, which has a hard time serializing even simple objects


def combine_choices(choices):
    """
    Explode a set of choices into possible combinations.

    Parameters
    ------------
    choices : :class:`dict`
        Dictionary where keys are names, and values are list of potential
        choices.

    Returns
    ---------
    :class:`list`
        List of dictionaries containing each set of choice combinations.

    Notes
    -----

        Will not append choices which are set to None.

        This requires Python 3.6+ (for ordered dictonaries).

    Todo
    ------

        Add option for coupled choices/restricted grids:

            X = [0, 1, 2], Y = [A, B, C] --> {X:0, Y:A}, {X:1, Y:B}, {X:2, Y:C}
    """
    if choices:  # if there are values specified
        index = np.array(
            np.meshgrid(*[np.arange(len(v)) for k, v in choices.items()])
        ).T.reshape(-1, len(choices))

        combs = []
        for ix in index:
            combs.append(
                {
                    k: v[vix]
                    for vix, (k, v) in zip(ix, choices.items())
                    if v[vix] is not None
                }
            )
        out = []
        for c in combs:  # don't duplicate configs
            if c not in out:
                out += [c]
        return out
    else:
        return [{}]


def func_wrapper(arg):
    func, kwargs = arg
    return func(**kwargs)


def multiprocess(func, param_sets):
    """
    Multiprocessing utility function, targeted towards large requests.
    Note that async is commonly slower for this use case.
    """
    jobs = [(func, params) for params in param_sets]
    with Pool(processes=len(jobs)) as p:
        results = p.map(func_wrapper, jobs)

    return results
