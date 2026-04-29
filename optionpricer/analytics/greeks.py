import numpy as np
from optionpricer.models.binomial import build_tree

def greeks(S, K, T, r, sigma, N=100, option_type="call", american=False):
    bump_sigma = 0.01
    bump_r     = 0.001
    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1 / u

    Su = S * u
    Sd = S * d

    fu  = build_tree(Su, K, T - dt,   r, sigma, N, option_type, american)
    fd  = build_tree(Sd, K, T - dt,   r, sigma, N, option_type, american)
    f   = build_tree(S,  K, T - dt,   r, sigma, N, option_type, american)
    fud = build_tree(S,  K, T - 2*dt, r, sigma, N, option_type, american)

    delta = (fu - fd) / (Su - Sd)
    gamma = ((fu - f) / (Su - S) - (f - fd) / (S - Sd)) / (0.5 * (Su - Sd))
    theta = (fud - f) / (2 * dt) / 365

    vega = (build_tree(S, K, T, r, sigma + bump_sigma, N, option_type, american)
          - build_tree(S, K, T, r, sigma - bump_sigma, N, option_type, american)) / (2 * bump_sigma)

    price_up   = build_tree(S, K, T, r + bump_r, sigma, N, option_type, american)
    price_down = build_tree(S, K, T, r - bump_r, sigma, N, option_type, american)
    rho = (price_up - price_down) / (2 * bump_r) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
