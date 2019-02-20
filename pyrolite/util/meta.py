def update_docstring_references(func, ref="ref"):
    """
    Updates docstring reference names to strings including the function name.
    Decorator will return the same function with a modified docstring. Sphinx
    likes unique names - specifically for citations, not so much for footnotes.
    """
    func.__doc__ = str(func.__doc__).replace(ref, func.__name__)
    return func
