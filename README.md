# OptionPricer

A high-performance quantitative option pricing library with Cython AOT compilation, Numba JIT, stochastic volatility models, and institutional-grade numerical precision.

## Installation

```bash
pip install optionpricer
```

For faster implied volatility (Jäckel's "Let's Be Rational"):
```bash
pip install optionpricer[fast]
```

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

## Pricing Models

| Model | Module | Style | Speed |
|:---|:---|:---|:---|
| Black-Scholes (erfc) | `black_scholes()` | European | ~0.18 ms |
| Binomial Tree (Cython/OpenMP) | `build_tree()` | American/European | ~0.3 ms (N=1000) |
| Crank-Nicolson FDM (Cython) | `crank_nicolson_fdm()` | American/European | ~3.3 ms (200×200) |
| Monte Carlo (Sobol + CV) | `monte_carlo_prices()` | All | ~1.0 ms (16K) |
| Merton Jump-Diffusion | `merton_jump_diffusion()` | European | ~0.15 ms |
| Carr-Madan FFT | `carr_madan_fft()` | European | ~0.23 ms |
| Heston Stochastic Vol | `heston_price()` | European | ~0.5 ms |
| Bates SVJD | `bates_price()` | European | ~0.5 ms |
| Quanto | `quanto_price()` | European | ~0.1 ms |
| Multi-Asset Basket (MC) | `basket_option()` | European | ~50 ms |
| Bjerksund-Stensland | `bjerksund_stensland_american()` | American | ~0.1 ms |
| Barrier (Analytical) | `barrier_analytical()` | European | ~0.1 ms |

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
