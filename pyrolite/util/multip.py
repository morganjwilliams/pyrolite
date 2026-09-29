from collections.abc import Callable
from multiprocessing import Pool
from typing import Any

import numpy as np

from .log import Handle

logger = Handle(__name__)


def combine_choices(choices: dict[str, list], include_none: bool = False) -> list[dict]:
    """
    Explode a set of choices into possible combinations.

    Parameters
    ------------
    choices : dict
        Dictionary where keys are names, and values are list of potential
        choices.
    include_none : bool
        Whether to include 'None' values, or otherwise omit them.

    Returns
    ---------
    list
        List of dictionaries containing each set of choice combinations.

    Notes
    -----

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
                    if ((v[vix] is not None) or include_none)
                }
            )
        out = []
        for c in combs:  # don't duplicate configs
            if c not in out:
                out += [c]
        return out
    else:
        return [{}]


def func_wrapper(arg: tuple[Any, dict]):
    func, kwargs = arg
    return func(**kwargs)


def multiprocess(func: Callable, param_sets: list[tuple[Any, dict]]):
    """
    Multiprocessing utility function, targeted towards large requests.
    Note that async is commonly slower for this use case.
    """
    jobs = [(func, params) for params in param_sets]
    with Pool(processes=len(jobs)) as p:
        results = p.map(func_wrapper, jobs)

    return results
