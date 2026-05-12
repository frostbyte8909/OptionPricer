# OptionPricer

A high-performance quantitative option pricing library with Cython AOT compilation, Numba JIT, stochastic volatility models, and institutional-grade numerical precision.

**Links:** [PyPI](https://pypi.org/project/optionpricer/) · [Source / issues](https://github.com/frostbyte8909/OptionPricer) · [Changelog](https://github.com/frostbyte8909/OptionPricer/blob/main/CHANGELOG.md)

**Versioning:** PyPI releases use `version` in [`pyproject.toml`](pyproject.toml) as the only source of truth. Every release Git tag must be **`v` + that version** (for example `0.2.1` in the file → tag `v0.2.1`). GitHub Actions checks this on tag push and again before publishing to PyPI. Details: [CONTRIBUTING.md](CONTRIBUTING.md).

## Installation

```bash
pip install optionpricer
```

PyPI currently publishes **source distributions** (`sdist`) only. Installing from PyPI compiles Cython extensions on your machine; you need a C compiler, Python headers, and **OpenMP** (see [Building extensions](#building-extensions)).

For faster implied volatility (Jäckel's "Let's Be Rational"):

```bash
pip install optionpricer[fast]
```

Optional developer and benchmark dependencies:

```bash
pip install optionpricer[dev]
```

(`dev` includes `pytest`, `psutil`, and `memory_profiler` used by the benchmark suite.)

## Development install

From a git clone:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

Run the test suite:

```bash
make test
# or: pytest tests/ -v
```

Contributor workflow (releases, PyPI OIDC, benchmarks): see [CONTRIBUTING.md](CONTRIBUTING.md).

## Building extensions

Extensions are defined in [setup.py](setup.py) (`_binomial_cy`, `_fdm_cy`). Editable install compiles them automatically when possible.

- **Linux:** Install a compiler toolchain (e.g. `build-essential` on Debian/Ubuntu). GCC’s `-fopenmp` links against `libgomp`.
- **macOS:** Install LLVM OpenMP via Homebrew, e.g. `brew install libomp`, so that include and library paths under `/opt/homebrew` or `/usr/local` match [setup.py](setup.py).

Create `sdist` / wheel locally:

```bash
pip install build
python -m build
```

Release automation uploads **`sdist` only** (no prebuilt cross-platform wheels yet). See [CONTRIBUTING.md](CONTRIBUTING.md) for the release checklist.

## gRPC API

Protobuf / gRPC stubs are generated from [optionpricer/api/pricer.proto](optionpricer/api/pricer.proto):

```bash
make grpc
```

Run the server (after installing `grpcio` and `grpcio-tools` if not already present):

```bash
python -m optionpricer.api.server
```

## Benchmarks and speed table

Wall times **depend on CPU, compiler, and load**. The table below lists **order-of-magnitude representative latencies** from a single reference run of [`tests/bench_v2.py`](tests/bench_v2.py). Reproduce on your machine:

```bash
make bench
# or: python tests/bench_v2.py
```

Machine-readable output (for CI artifacts or tooling):

```bash
python tests/bench_v2.py --json
```

**Mapping:** where a **Bench kernel** name appears, it matches a row in the benchmark report. Other models are not in the automated suite; figures are indicative only.

| Model | Module | Style | Rep. latency* | Bench kernel |
|:---|:---|:---|:---|:---|
| Black-Scholes (erfc) | `black_scholes()` | European | ~0.18 ms | BSM scalar |
| Binomial Tree (Cython/OpenMP) | `build_tree()` | American/European | ~0.3 ms (N=1000) | Binomial N=1000 |
| Crank-Nicolson FDM (Cython) | `crank_nicolson_fdm()` | American/European | ~3 ms (200×200 grid) | FDM 200x200 |
| Monte Carlo (Sobol + CV) | `monte_carlo_prices()` | European vanilla | ~1 ms (16K paths) | MC 16K paths |
| Merton Jump-Diffusion | `merton_jump_diffusion()` | European | ~0.15 ms | Merton JD |
| Carr-Madan FFT | `carr_madan_fft()` | European | ~0.23 ms | FFT 4096 |
| Heston Stochastic Vol | `heston_price()` | European | ~0.5 ms | — |
| Bates SVJD | `bates_price()` | European | ~0.5 ms | — |
| Quanto | `quanto_price()` | European | ~0.1 ms | — |
| Multi-Asset Basket (MC) | `basket_option()` | European | ~50 ms | — |
| Bjerksund-Stensland | `bjerksund_stensland_american()` | American | ~0.1 ms | — |
| Barrier (Analytical) | `barrier_analytical()` | European | ~0.1 ms | — |

\*Median wall-clock ms from `bench_v2` where listed; otherwise typical single-call order of magnitude on a laptop-class CPU.

## Quick Start

```python
from optionpricer import OptionContract, MarketState, black_scholes, build_tree

contract = OptionContract(strike=100, expiry=1.0, option_type="call")
market = MarketState(spot=100, rate=0.05, volatility=0.2)

# Analytical BSM (erfc-based, machine precision at ±8σ tails)
price = black_scholes(contract, market)

# American put via Cython binomial tree
put = OptionContract(strike=100, expiry=1.0, option_type="put", american=True)
american_price = build_tree(put, market, N=5000)

# Full Greek vector via closed-form AAD
from optionpricer import aad_greeks
greeks = aad_greeks(contract, market)
# -> {delta, gamma, vega, theta, rho, vanna, volga, charm}
```

## Analytics

| Module | Function | Description |
|:---|:---|:---|
| Implied Volatility | `implied_vol()` | Jäckel → Newton → Brent fallback |
| Greeks (BSM) | `greeks()` | Closed-form Euro, FD bumping American |
| AAD Greeks | `aad_greeks()` | Δ, Γ, V, Θ, ρ, Vanna, Volga, Charm |
| Malliavin MC Greeks | `malliavin_greeks()` | Smooth Greeks for non-diff payoffs |
| GARCH(1,1) | `garch_fit()` | MLE calibration with Numba variance path |
| EWMA Vol | `ewma_volatility()` | 30-day fallback estimator |
| SABR Smile | `sabr_implied_vol()` | Hagan formula with edge guards |
| Dupire Local Vol | `dupire_local_vol()` | Surface from IV grid via spline |
| Vanna-Volga | `vanna_volga_price()` | Smile-adjusted exotic pricing |
| Nelson-Siegel | `fit_nelson_siegel()` | Yield curve calibration |
| Arbitrage Check | `arbitrage_check()` | Calendar + butterfly validation |

## Architecture

- **Cython AOT:** Binomial tree and FDM solvers compiled to C with OpenMP for multi-core parallelism
- **Numba JIT:** Monte Carlo paths, GARCH variance loop, Malliavin kernels compiled via LLVM
- **erfc precision:** All BSM CDF calls use `erfc` to maintain accuracy at extreme tails
- **Pydantic V2:** Type-safe contract/market schemas with validation

## License

Apache 2.0 — see [LICENSE](LICENSE).
