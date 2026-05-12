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

## Release checklist (Git tag = PyPI version)

**Single source of truth:** `version` in [`pyproject.toml`](pyproject.toml). The file on the tagged commit is what becomes the PyPI release.

1. Set `version = "X.Y.Z"` in `pyproject.toml` and document it in [`CHANGELOG.md`](CHANGELOG.md).
2. Commit and push to `main`.
3. **Before** creating the GitHub Release, verify locally (optional but recommended):

   ```bash
   make check-version-tag TAG=vX.Y.Z
   ```

   Example: if `pyproject.toml` says `0.2.1`, the tag **must** be exactly `v0.2.1`.

4. Create a **GitHub Release** from tag `vX.Y.Z` (same string). Publishing the release runs [`.github/workflows/release.yml`](.github/workflows/release.yml), which **aborts** if the tag and `pyproject.toml` disagree, then uploads the `sdist` to PyPI via Trusted Publishing.

5. Pushing tag `v*` also runs CI [`.github/workflows/ci.yml`](.github/workflows/ci.yml) `verify-release-tag` so mismatches fail before you even open a Release.

6. Confirm the new version on [PyPI](https://pypi.org/project/optionpricer/).

### PyPI Trusted Publishing

In the PyPI project settings, add a **trusted publisher** for GitHub:

- Repository: `frostbyte8909/OptionPricer`
- Workflow: `release.yml`
- Environment: leave blank unless you use a GitHub Environment named in the workflow

No long-lived `PYPI_API_TOKEN` is required when OIDC is configured. If you rename the workflow file, update the trusted publisher entry on PyPI accordingly.
