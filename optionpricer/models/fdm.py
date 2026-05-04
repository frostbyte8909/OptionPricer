import numpy as np
from typing import Union
from optionpricer.core import OptionContract, MarketState

try:
    from optionpricer.models._fdm_cy import _crank_nicolson_psor_vectorized
except ImportError:
    raise ImportError("Cython extension _fdm_cy not found. Please compile the package.")

def crank_nicolson_fdm(
    contract: OptionContract, 
    market: MarketState, 
    M: int = 400, 
    N: int = 400, 
    omega: float = 1.2, 
    tol: float = 1e-6, 
    max_iter: int = 1000
) -> Union[float, np.ndarray]:
    """Price an option via Crank-Nicolson finite differences with PSOR.

    Uses a Cython-compiled kernel for the Projected Successive
    Over-Relaxation iterations required for American early exercise.

    Args:
        contract: Option contract (supports American exercise).
        market: Market state.
        M: Number of spatial grid points.
        N: Number of time steps.
        omega: SOR relaxation parameter (1.0 < omega < 2.0).
        tol: Convergence tolerance for PSOR iterations.
        max_iter: Maximum PSOR iterations per time step.

    Returns:
        Option price as float or ndarray.
    """
    b = np.broadcast(market.spot, contract.strike, contract.expiry, market.rate, market.volatility, market.dividend)
    
    S_arr = np.broadcast_to(market.spot, b.shape).astype(float).ravel()
    K_arr = np.broadcast_to(contract.strike, b.shape).astype(float).ravel()
    T_arr = np.broadcast_to(contract.expiry, b.shape).astype(float).ravel()
    r_arr = np.broadcast_to(market.rate, b.shape).astype(float).ravel()
    sigma_arr = np.broadcast_to(market.volatility, b.shape).astype(float).ravel()
    q_arr = np.broadcast_to(market.dividend, b.shape).astype(float).ravel()
    
    result_arr = np.empty_like(S_arr)
    
    _crank_nicolson_psor_vectorized(
        S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr, result_arr,
        int(M), int(N), contract.option_type == "call", contract.american,
        float(omega), float(tol), int(max_iter)
    )
    
    result = result_arr.reshape(b.shape)
    if result.ndim == 0:
        return float(result.item())
    return result
