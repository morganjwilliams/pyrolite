# Development

## Development History and Planning

- [Changelog](changelog.md)
- [Roadmap](roadmap.md)

## Contributing

- [Contributing](contributing.md)
- [Contributors](contributors.md)
- [Code of Conduct](conduct.md)

## Development Installation

To access and use the development version, you can
[clone the repository](https://github.com/morganjwilliams/pyrolite) and
set up the environment:

```bash
git clone https://github.com/morganjwilliams/pyrolite.git
git checkout develop
uv sync --extra dev
```

## Tests

If you clone the source repository, unit tests can be run using pytest from the root
directory after installation with development dependencies
({code}`pip install -e .[dev]`):

```bash
uv run pytest
```

If instead you only want to test a subset, you can call {mod}`pytest` directly from
within the pyrolite repository:

```bash
uv run pytest ./test/<path to test or test folder>
```
