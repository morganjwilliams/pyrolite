from numpydoc.docscrape import FunctionDoc, ClassDoc
import webbrowser
import inspect

def take_me_to_the_docs():
    webbrowser.open('https://pyrolite.rtfd.io')

def sphinx_doi_link(doi):
    return "`{} <https://dx.doi.org/{}>`__".format(doi, doi)


def subkwargs(kwargs, *f):
    """
    Get a subset of keyword arguments which are accepted by a function.

    Parameters
    ----------
    kwargs : :class:`dict`
        Dictionary of keyword arguments.
    f : :class:`callable`
        Function(s) to check.
    """
    return {k: v for k, v in kwargs.items() if inargs(k, *f)}


def inargs(name, *f):
    """
    Check if an argument is a possible input for a specific function.

    Parameters
    ----------
    name : :class:`str`
        Argument name.
    f : :class:`callable`
        Function(s) to check.
    """
    args = []
    for f in f:
        args += inspect.getfullargspec(f).args
    return name in args


def numpydoc_str_param_list(iterable, indent=4):
    """
    Parameters
    -------------
    iterable : :class:`list`
        List of numpydoc parameters.
    indent : :class:`int`
        Indent as number of spaces.

    Returns
    -------
    :class:`str`
    """
    out = []
    for param in iterable:
        if param[1]:
            out += ["%s : %s" % (param[0], param[1])]
        else:
            out += [param[0]]
        if param[2] and "".join(param[2]).strip():
            out += [indent * " " + i for i in param[2]]
    out += [""]
    return ("\n" + indent * " ").join(out)


def get_additional_params(
    *fs, t="Parameters", header="", indent=4, subsections=False, subsection_delim="Note"
):
    """
    Checks the base Parameters section of docstrings to get 'Other Parameters'
    for a specific function. Designed to incorporate information on inherited
    or forwarded parameters.

    Parameters
    -------------
    fs : :class:`list`
        List of functions.
    t : :class:`str`
        Target block of docstrings.
    header : :class:`str`
        Optional seciton header.
    indent : :class:`int` | :class:`str`
        Indent as number of spaces, or a string of a given length.
    subsections : :class:`bool`, `False`
        Whether to include headers specific for each function, creating subsections.
    subsection_delim : :class:`str`
        Subsection delimiter.

    Returns
    --------
    :class:`str`

    Todo
    --------
        * Add delimiters between functions to show where arguments should be passed.
    """
    if isinstance(indent, str):
        indent = len(indent)

    if header:
        sectionheader = [header, "-" * (len(header) + 1)]
    else:
        sectionheader = []

    docs = [(f, FunctionDoc(f)) for f in fs]
    pars = []
    subsects = []
    p0 = [i[0] for i in docs[0][1][t]]
    for (f, d) in docs[1:]:  # add things which haven't already been registered
        new = [o for o in d[t] if not (o[0] in p0 or o[0] in pars)]
        if subsections:
            subsection = numpydoc_str_param_list(new, indent=indent)
            subsection = ("\n" + " " * indent) + ("\n" + " " * indent).join(
                [
                    ("\n" + " " * indent).join(
                        [subsection_delim, "-" * (len(subsection_delim) + 1)]
                    )
                ]
                + [
                    "The following additional parameters are from :func:`{}`.".format(
                        ".".join([f.__module__, f.__name__])
                    )
                ]
                + [("\n" + " " * indent).join([header, "-" * (len(header) + 1)])]
                + [subsection]
            )
            subsects.append(subsection)
        else:
            pars += new

    if not subsections:
        section = numpydoc_str_param_list(pars, indent=indent)
        section = ("\n" + " " * indent).join(sectionheader + [section])
    else:
        section = ("\n" + " " * indent).join(subsects)
    return section


def update_docstring_references(func, ref="ref"):
    """
    Updates docstring reference names to strings including the function name.
    Decorator will return the same function with a modified docstring. Sphinx
    likes unique names - specifically for citations, not so much for footnotes.
    """
    func.__doc__ = str(func.__doc__).replace(ref, func.__name__)
    return func
