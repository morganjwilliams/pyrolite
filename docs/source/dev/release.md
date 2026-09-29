# Release Guide

Before releasing, ensure that the changelog is up to date,
consider preloading the next release link, and ensure the `uv`
lock is up to date:

```bash
uv sync --extra dev
git add uv.lock
git commit "Update uv Lock"
```

The project can be build into distributable artefacts with:

```bash
uv run python -m build --sdist --wheel
```

We can check this distribution with `twine`:

```bash
uv run twine check ./dist/*
```

And, assuming credentials exsit, uploaded to PyPI with:

```bash
uv run twine upload ./dist/*
```
