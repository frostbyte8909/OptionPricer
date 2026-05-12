# Changelog

## v0.2.2

[+] `scripts/verify_release_version.py` — Git tag `v*` must match `project.version` in `pyproject.toml`
[+] CI: verify-release-tag job on tag push; release workflow fails publish if tag and version disagree
[+] `make check-version-tag TAG=vX.Y.Z` for local preflight; README / CONTRIBUTING versioning notes
[+] Project URLs default to `frostbyte8909/optionpricer` on GitHub

## v0.2.1

[+] README: development install, build/grpc, PyPI links, and benchmark methodology for the speed table
[+] CONTRIBUTING.md with tests, benchmarks, local build, and PyPI release (OIDC) checklist
[+] `make bench` target; `tests/bench_v2.py --json` and `--smoke` CLI flags
[+] Pytest smoke tests for benchmark harness (finite latency, no crash)
[+] GitHub Actions CI (Ubuntu, Python 3.10 / 3.12) with editable install and pytest
[+] GitHub Actions release workflow publishing **sdist** to PyPI via Trusted Publishing
[+] `pyproject.toml`: PyPI/Homepage/Issues URLs, `pytest` markers, `build`/`twine` in `dev` extras

## v0.2.0

[+] Heston stochastic volatility model (precision)
[+] Bates SVJD model (precision)
[+] Quanto option pricing (coverage)
[+] Multi-asset basket/spread pricing via Cholesky MC (coverage)
[+] Nelson-Siegel/Svensson term structure models (coverage)
[+] GARCH(1,1) MLE volatility calibration (speed)
[+] SABR Hagan smile model with edge-case guards (precision)
[+] Dupire local volatility surface (coverage)
[+] Carr-Madan FFT pricing with pluggable CFs (speed)
[+] Merton Jump-Diffusion model (coverage)
[+] AAD closed-form Greeks: Vanna, Volga, Charm (speed)
[+] Malliavin calculus MC Greeks (precision)
[+] Vanna-Volga 3-point exotic pricing (coverage)
[+] EWMA volatility fallback (reliability)
[+] Arbitrage-free surface validation (reliability)
[+] Contract-model compatibility guardrails (reliability)
[+] MC chunked path processing (efficiency)
[+] erfc-based CDF across all BSM kernels (precision)
[+] Comprehensive test suite: 30+ tests (reliability)
[+] Apache 2.0 license (compliance)
[+] Google-style docstrings on all public functions (docs)

[-] scipy.stats.norm.cdf in BSM core (outdated, precision)
[-] Positional MarketState construction in greeks.py (outdated, broken)
[-] Broken validator string comparison in core.py (outdated, bugfix)

## v0.1.11

[+] Cython binomial tree with OpenMP (speed)
[+] Cython FDM PSOR solver (speed)
[+] Numba Monte Carlo with Sobol QMC (speed)
[+] Control variates variance reduction (efficiency)
[+] Bjerksund-Stensland American approximation (coverage)
[+] Analytical barrier option pricing (coverage)
[+] CLI interface (usability)
[+] Pydantic V2 contract/market schemas (reliability)
