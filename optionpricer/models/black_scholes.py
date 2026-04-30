import numpy as np
from scipy.stats import norm

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Calculate the theoretical price of a European option using the Black-Scholes model.

    Args:
        S (float): Current asset price.
        K (float): Strike price of the option.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying asset (annualized).
        option_type (str, optional): 'call' for Call option, 'put' for Put option. Defaults to 'call'.

    Returns:
        float: The theoretical price of the option.
        
    Raises:
        ValueError: If option_type is not 'call' or 'put'.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df = np.exp(-r * T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * df * norm.cdf(d2)
    elif option_type == "put":
        return K * df * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

if __name__ == "__main__":
    from optionpricer.analytics.greeks import greeks

    print("=== Options Greeks Calculator ===\n")
    S     = float(input("Stock price (S):       "))
    K     = float(input("Strike price (K):      "))
    T     = float(input("Time to expiry (years): "))
    r     = float(input("Risk-free rate (0.05 = 5%): "))
    sigma = float(input("Volatility (0.2 = 20%): "))
    opt   = input("Option type (call/put): ").strip().lower()

    result = greeks(S=S, K=K, T=T, r=r, sigma=sigma, option_type=opt)
    print("\n--- Greeks ---")
    for key, val in result.items():
        print(f"  {key.capitalize():<8} {float(val):.6f}")

