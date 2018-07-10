from pathos.multiprocessing import ProcessingPool as Pool

# Note : Using pathos multiprocessing which leverages dill over standard
# pickle, which has a hard time serializing even simple objects

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
