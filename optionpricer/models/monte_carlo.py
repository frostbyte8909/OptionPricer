import numpy as np
from scipy.stats.qmc import Sobol
from scipy.stats import norm


def payoff(ST: np.ndarray, K: float, option_type: str = "call") -> np.ndarray:
    if option_type=="call":
        return np.maximum(ST - K, 0) 
    else:
        return np.maximum(K - ST, 0)

def monte_carlo_prices(S: float, K: float, T: float, r: float, sigma: float, N: int = 32_768, option_type: str = "call") -> float:
    """
    Price a European option using Monte Carlo simulation.
    Features Sobol sequences (quasi-Monte Carlo), antithetic variates, and control variates for variance reduction.

    Args:
        S (float): Current asset price.
        K (float): Strike price of the option.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying asset (annualized).
        N (int, optional): Number of simulation paths (rounded up to nearest power of 2). Defaults to 32,768.
        option_type (str, optional): 'call' for Call option, 'put' for Put option. Defaults to 'call'.

    Returns:
        float: The theoretical price of the option.
    """
    half = N // 2
    half = 1 << (half - 1).bit_length()

    sampler = Sobol(d=1, scramble=True)
    Z = norm.ppf(np.clip(sampler.random(half), 1e-10, 1 - 1e-10)).ravel()

    drift  = (r - 0.5 * sigma**2) * T
    vol    = sigma * np.sqrt(T)
    
    ST_all = np.empty(N)
    ST_all[:half] = S * np.exp(drift + vol * Z)
    ST_all[half:] = S * np.exp(drift - vol * Z)

    p_all = np.empty(N)
    if option_type == "call":
        np.maximum(ST_all - K, 0, out=p_all)
    else:
        np.maximum(K - ST_all, 0, out=p_all)

    E_ST   = S * np.exp(r * T)
    ST_dev = ST_all - ST_all.mean()
    p_mean = p_all.mean()
    
    c = -np.dot(p_all - p_mean, ST_dev) / np.dot(ST_dev, ST_dev)

    return np.exp(-r * T) * (p_all + c * (ST_all - E_ST)).mean()


def convergence_test(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> None:
    sizes = [100, 1000, 10_000, 100_000, 1_000_000]
    for N in sizes:
        price = monte_carlo_prices(S, K, T, r, sigma, N, option_type)
        print(f"N={N:<10} price={price:.4f}")

if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    print("=== Monte Carlo Pricer ===\n")
    print(f"Single estimate (N=32,768): {monte_carlo_prices(S, K, T, r, sigma):.4f}")
    print(f"\nB-S analytical price should be ~10.4506\n")
    print("=== Convergence Test ===")
    convergence_test(S, K, T, r, sigma)