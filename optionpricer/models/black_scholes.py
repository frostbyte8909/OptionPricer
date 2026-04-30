import numpy as np
from scipy.stats import norm
from typing import Union

def black_scholes(S: Union[float, np.ndarray], K: Union[float, np.ndarray], T: Union[float, np.ndarray], r: Union[float, np.ndarray], sigma: Union[float, np.ndarray], q: Union[float, np.ndarray] = 0.0, option_type: str = "call") -> Union[float, np.ndarray]:
    """
    Calculate the theoretical price of a European option using the Black-Scholes-Merton model.
    Supports continuous dividend yields and vectorized inputs (NumPy arrays).
    Handles edge cases for expiration (T=0) and deterministic states (sigma=0).

    Args:
        S (Union[float, np.ndarray]): Current asset price.
        K (Union[float, np.ndarray]): Strike price of the option.
        T (Union[float, np.ndarray]): Time to maturity in years.
        r (Union[float, np.ndarray]): Risk-free interest rate (annualized).
        sigma (Union[float, np.ndarray]): Volatility of the underlying asset (annualized).
        q (Union[float, np.ndarray], optional): Continuous dividend yield. Defaults to 0.0.
        option_type (str, optional): 'call' for Call option, 'put' for Put option. Defaults to 'call'.

    Returns:
        Union[float, np.ndarray]: The theoretical price of the option(s).
        
    Raises:
        ValueError: If option_type is not 'call' or 'put'.
    """
    S_arr = np.atleast_1d(S).astype(float)
    K_arr = np.atleast_1d(K).astype(float)
    T_arr = np.atleast_1d(T).astype(float)
    r_arr = np.atleast_1d(r).astype(float)
    sigma_arr = np.atleast_1d(sigma).astype(float)
    q_arr = np.atleast_1d(q).astype(float)

    is_expired = T_arr <= 1e-8
    is_deterministic = sigma_arr <= 1e-8
    mask_edge = is_expired | is_deterministic

    result = np.zeros_like(S_arr, dtype=float)

    if np.any(mask_edge):
        if option_type == "call":
            result[mask_edge] = np.maximum(S_arr[mask_edge] * np.exp(-q_arr[mask_edge] * T_arr[mask_edge]) - K_arr[mask_edge] * np.exp(-r_arr[mask_edge] * T_arr[mask_edge]), 0.0)
        elif option_type == "put":
            result[mask_edge] = np.maximum(K_arr[mask_edge] * np.exp(-r_arr[mask_edge] * T_arr[mask_edge]) - S_arr[mask_edge] * np.exp(-q_arr[mask_edge] * T_arr[mask_edge]), 0.0)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    mask_calc = ~mask_edge
    if np.any(mask_calc):
        S_c = S_arr[mask_calc]
        K_c = K_arr[mask_calc]
        T_c = T_arr[mask_calc]
        r_c = r_arr[mask_calc]
        sig_c = sigma_arr[mask_calc]
        q_c = q_arr[mask_calc]

        d1 = (np.log(S_c / K_c) + (r_c - q_c + 0.5 * sig_c**2) * T_c) / (sig_c * np.sqrt(T_c))
        d2 = d1 - sig_c * np.sqrt(T_c)
        
        df_r = np.exp(-r_c * T_c)
        df_q = np.exp(-q_c * T_c)

        if option_type == "call":
            result[mask_calc] = S_c * df_q * norm.cdf(d1) - K_c * df_r * norm.cdf(d2)
        elif option_type == "put":
            result[mask_calc] = K_c * df_r * norm.cdf(-d2) - S_c * df_q * norm.cdf(-d1)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    if result.size == 1:
        return float(result.item())
    return result
