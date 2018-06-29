from multiprocessing import Process, Pool


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
        # This is already an async map within multiprocessing
        results = p.map(func_wrapper, jobs)

    return results
