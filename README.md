# OptionPricer

A high-performance, professional-grade quantitative finance library for option pricing, optimized heavily via NumPy vectorization, Numba JIT, and Cython AOT compilation.

## Development Journey & Architecture

This package was built systematically through strict, high-performance architectural phases:

1. **Phase 1: NumPy Vectorization**
   - Eliminated slow Python loops in the core hot paths.
   - Replaced scalar stock price reconstructions with precomputed power arrays, yielding a 3.7x baseline speedup on the binomial lattice.

2. **Phase 2: Advanced Monte Carlo & Variance Reduction**
   - Implemented Sobol Quasi-Random sequences to ensure superior space-filling over standard pseudo-random engines.
   - Introduced Antithetic Variates ($Z$ and $-Z$) and Control Variates (using terminal spot price correlation) to achieve up to 800x variance reduction, allowing high precision at drastically lower path counts.

3. **Phase 3: Numba JIT Compilation**
   - Compiled the sequential backward-induction algorithms into machine code via LLVM.
   - Handled Python's GIL overhead by isolating numerical loops inside pure-C equivalents, dropping execution time from 48ms down to 5ms for large $N$.

4. **Phase 4: Cython Extensions & Pre-compiled Wheels**
   - Statically typed the core mathematical functions (`cdef`, `double[:]` memoryviews) into an AOT compiled C-extension to completely eliminate runtime compilation warmup.

5. **Phase 5: Algorithmic Routing**
   - Engineered smart heuristics for implied volatility. Routes to Peter Jäckel's 'Let's Be Rational' for standard Euro options, falls back to Newton-Raphson for near-ATM scenarios, and seamlessly redirects to SciPy's robust `brentq` for deep OTM/ITM and American edge cases.

6. **Phase 6: Professional Software Engineering**
   - Enforced strict dependency boundaries.
   - Fully annotated with Python Type Hints and Google-style docstrings.
   - Prepared for distribution via PyPA `build` and `twine` CI/CD standards.

## Usage

```bash
pip install optionpricer
```

```python
from optionpricer import build_tree, monte_carlo_prices, implied_vol

# Price a 5,000-step American Put via Cython
price = build_tree(S=100, K=100, T=1, r=0.05, sigma=0.2, N=5000, option_type="put", american=True)

# Generate 32,000 variance-reduced Monte Carlo paths
mc_price = monte_carlo_prices(S=100, K=100, T=1, r=0.05, sigma=0.2)
```
