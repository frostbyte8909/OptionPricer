import numpy as np


def simulate_ST(S, T, r, sigma, N):
    Z = np.random.standard_normal (N)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return ST

def payoff(ST, K, option_type="call"):
    if option_type=="call":
        return np.maximum(ST - K, 0) 
    else:
        return np.maximum(K - ST, 0)

def monte_carlo_prices(S, K, T, r, sigma, N=100_000, option_type="call"):
    ST = simulate_ST(S, T, r, sigma, N)
    payoffs = payoff(ST, K, option_type)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price

def convergence_test(S, K, T, r, sigma,  option_type="call"):
    sizes = [100, 1000, 10_000, 100_000, 1_000_000]
    for N in sizes:
        price = monte_carlo_prices(S, K, T, r, sigma, N, option_type)
        print(f"N={N:<10} price={price:.4f}")

if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    print("=== Monte Carlo Pricer ===\n")
    print(f"Single estimate (N=100,000): {monte_carlo_prices(S, K, T, r, sigma):.4f}")
    print(f"\nB-S analytical price should be ~10.4506\n")
    print("=== Convergence Test ===")
    convergence_test(S, K, T, r, sigma)