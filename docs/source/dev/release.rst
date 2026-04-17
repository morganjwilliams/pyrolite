Release Guide
==============

Before releasing, ensure that the changelog is up to date,
consider preloading the next release link, and ensure the `uv`
lock is up to date:

.. code-block:: bash
    
    uv sync --extra dev
    git add uv.lock
    git commit "Update uv Lock"


The project can be build into distributable artefacts with:

.. code-block:: bash
    
    uv run python -m build --sdist --wheel

We can check this distribution with `twine`:

.. code-block:: bash
    
    uv run twine check ./dist/*

And, assuming credentials exsit, uploaded to PyPI with:

.. code-block:: bash
    
    uv run twine upload ./dist/*