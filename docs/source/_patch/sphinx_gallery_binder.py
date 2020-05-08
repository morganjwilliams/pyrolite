from sphinx_gallery import binder

def alt_gen_binder_rst(
    fpath, binder_conf, gallery_conf, img="https://mybinder.org/badge_logo.svg"
):
    """Generate the RST + link for the Binder badge.
    Parameters
    ----------
    fpath: str
        The path to the `.py` file for which a Binder badge will be generated.
    binder_conf: dict or None
        If a dictionary it must have the following keys:
        'binderhub_url'
            The URL of the BinderHub instance that's running a Binder service.
        'org'
            The GitHub organization to which the documentation will be pushed.
        'repo'
            The GitHub repository to which the documentation will be pushed.
        'branch'
            The Git branch on which the documentation exists (e.g., gh-pages).
        'dependencies'
            A list of paths to dependency files that match the Binderspec.
    Returns
    -------
    rst : str
        The reStructuredText for the Binder badge that links to this file.
    """
    binder_conf = binder.check_binder_conf(binder_conf)
    binder_url = binder.gen_binder_url(fpath, binder_conf, gallery_conf)
    rst = (
        "\n" ".. image:: {0}\n" "    :target: {1}\n" "    :alt: Launch Binder\n"
    ).format(img, binder_url)
    return rst

binder.gen_binder_rst = alt_gen_binder_rst
