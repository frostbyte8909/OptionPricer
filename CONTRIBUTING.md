# Contributing to OptionPricer

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

See the [README](README.md) for compiler and OpenMP requirements when building Cython extensions.

## Tests

```bash
make test
# or
pytest tests/ -v
```

Fast benchmark smoke checks (finite timing, no crash) live in `tests/test_bench_smoke.py` and run with the default suite.

## Benchmarks

Full institutional benchmark report (median / p99 / memory):

```bash
make bench
# or
python tests/bench_v2.py
```

JSON only (stdout, suitable for CI artifacts):

```bash
python tests/bench_v2.py --json
```

Minimal iterations (local or CI smoke of the whole suite):

```bash
python tests/bench_v2.py --smoke --json
```

## gRPC stubs

After editing `optionpricer/api/pricer.proto`:

```bash
make grpc
```

Install `grpcio` and `grpcio-tools` in your environment if missing.

## Local packaging

```bash
python -m build
```

Inspect `dist/` (`sdist` and a platform wheel may be produced). Release automation uploads **`sdist` only** to PyPI.

## Release checklist

1. Bump `version` in [`pyproject.toml`](pyproject.toml).
2. Update [`CHANGELOG.md`](CHANGELOG.md) for the new version.
3. Run `make test` and optionally `make bench`.
4. Commit and push; create a **GitHub Release** (tag `vX.Y.Z` matching the version). The [Publish workflow](.github/workflows/release.yml) runs on `release: published` and uploads the `sdist` to PyPI via **Trusted Publishing (OIDC)**.
5. Confirm the release on [PyPI](https://pypi.org/project/optionpricer/).

### PyPI Trusted Publishing

In the PyPI project settings, add a **trusted publisher** for GitHub:

- Repository: `frostbyte8909/optionpricer`
- Workflow: `release.yml`
- Environment: leave blank unless you use a GitHub Environment named in the workflow

No long-lived `PYPI_API_TOKEN` is required when OIDC is configured. If you rename the workflow file, update the trusted publisher entry on PyPI accordingly.
