from optionpricer.models.black_scholes import black_scholes
from optionpricer.models.binomial import build_tree
from optionpricer.analytics.greeks import greeks
from optionpricer.core import OptionContract, MarketState

if __name__ == "__main__":
    S     = float(input("Spot price (S): "))
    K     = float(input("Strike price (K): "))
    T     = float(input("Time to expiry in years (T): "))
    r     = float(input("Risk-free rate (r, e.g. 0.05): "))
    q     = float(input("Dividend yield (q, e.g. 0.0): ") or 0.0)
    sigma = float(input("Volatility (sigma, e.g. 0.2): "))
    N     = int(input("Number of steps (N, default 100): ") or 100)

    market = MarketState(spot=S, rate=r, volatility=sigma, dividend=q)
    
    bs_price  = black_scholes(OptionContract(K, T, "call"), market)
    euro_call = build_tree(OptionContract(K, T, "call", False), market, N=N)
    amer_call = build_tree(OptionContract(K, T, "call", True), market, N=N)
    euro_put  = build_tree(OptionContract(K, T, "put", False), market, N=N)
    amer_put  = build_tree(OptionContract(K, T, "put", True), market, N=N)
    
    g = greeks(OptionContract(K, T, "call"), market, N=N)

    print(f"\nB-S analytical price:            {bs_price:.4f}")
    print(f"European Call (Binomial, N={N}): {euro_call:.4f}")
    print(f"American Call (Binomial, N={N}): {amer_call:.4f}")
    print(f"European Put  (Binomial, N={N}): {euro_put:.4f}")
    print(f"American Put  (Binomial, N={N}): {amer_put:.4f}")
    print(f"Difference (Euro Call vs B-S):   {abs(euro_call - bs_price):.6f}")
    print(f"\n--- Greeks (European Call) ---")
    for name, val in g.items():
        print(f"  {name:>6}: {val:.6f}")
