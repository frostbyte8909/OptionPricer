import numpy as np
from optionpricer.models.binomial import build_tree

def greeks(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, N: int = 100, option_type: str = "call", american: bool = False) -> dict[str, float]:
    """
    Calculate the options Greeks (Delta, Gamma, Theta, Vega, Rho) natively.
    Uses exact analytical derivatives for European options and finite difference on the binomial tree for American options.
    Supports continuous dividend yields.

    Args:
        S (float): Current asset price.
        K (float): Strike price of the option.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying asset (annualized).
        q (float, optional): Continuous dividend yield. Defaults to 0.0.
        N (int, optional): Number of steps in the binomial tree used for pricing American options. Defaults to 100.
        option_type (str, optional): 'call' for Call option, 'put' for Put option. Defaults to 'call'.
        american (bool, optional): If True, calculates Greeks for an American option. Defaults to False.

    Returns:
        dict[str, float]: A dictionary containing 'delta', 'gamma', 'theta', 'vega', and 'rho'.
    """
    if not american:
        from scipy.stats import norm
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0
        
        delta = np.exp(-q * T) * cdf_d1
        gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)
        
        if option_type == "call":
            theta = (- (S * sigma * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T)) + q * S * np.exp(-q * T) * norm.cdf(d1) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            theta = (- (S * sigma * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T)) - q * S * np.exp(-q * T) * norm.cdf(-d1) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

    bump_sigma = 0.01
    bump_r     = 0.001
    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1 / u

    Su = S * u
    Sd = S * d

    fu  = build_tree(Su, K, T - dt,   r, sigma, q, N, option_type, american)
    fd  = build_tree(Sd, K, T - dt,   r, sigma, q, N, option_type, american)
    f   = build_tree(S,  K, T - dt,   r, sigma, q, N, option_type, american)
    fud = build_tree(S,  K, T - 2*dt, r, sigma, q, N, option_type, american)

    delta = (fu - fd) / (Su - Sd)
    gamma = ((fu - f) / (Su - S) - (f - fd) / (S - Sd)) / (0.5 * (Su - Sd))
    theta = (fud - f) / (2 * dt) / 365

    vega = (build_tree(S, K, T, r, sigma + bump_sigma, q, N, option_type, american)
          - build_tree(S, K, T, r, sigma - bump_sigma, q, N, option_type, american)) / (2 * bump_sigma)

    price_up   = build_tree(S, K, T, r + bump_r, sigma, q, N, option_type, american)
    price_down = build_tree(S, K, T, r - bump_r, sigma, q, N, option_type, american)
    rho = (price_up - price_down) / (2 * bump_r) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
