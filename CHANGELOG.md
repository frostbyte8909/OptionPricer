# Changelog

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
