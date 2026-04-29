import numpy as np
from scipy.stats import norm
from __pycache__.Models.greeks import greeks

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

if __name__ == "__main__":
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

