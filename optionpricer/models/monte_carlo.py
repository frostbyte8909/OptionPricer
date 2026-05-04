import numpy as np
from scipy.stats.qmc import Sobol
from scipy.stats import norm
from numba import njit, prange
from typing import Union
from optionpricer.core import OptionContract, MarketState, ExoticType, OptionType
from optionpricer.models.black_scholes import black_scholes

@njit(parallel=True, fastmath=True, cache=True)
def _generate_paths_brownian_bridge(S0_arr, r_arr, q_arr, sig_arr, T_arr, Z, N_steps, N_paths):
    num_options = S0_arr.shape[0]
    dt = T_arr / N_steps
    
    paths = np.empty((num_options, N_paths, N_steps + 1))
    
    for k in prange(num_options):
        S0 = S0_arr[k]
        drift = (r_arr[k] - q_arr[k] - 0.5 * sig_arr[k]**2) * dt[k]
        vol = sig_arr[k] * np.sqrt(dt[k])
        
        for p in range(N_paths):
            paths[k, p, 0] = S0
            for t in range(1, N_steps + 1):
                paths[k, p, t] = paths[k, p, t-1] * np.exp(drift + vol * Z[p, t-1])
                
    return paths

@njit(parallel=True, fastmath=True, cache=True)
def _price_exotic_numba(paths, K_arr, r_arr, T_arr, exotic_type_id, is_call, barrier, rebate):
    num_options = paths.shape[0]
    N_paths = paths.shape[1]
    N_steps = paths.shape[2] - 1
    
    result = np.empty(num_options)
    
    for k in prange(num_options):
        df = np.exp(-r_arr[k] * T_arr[k])
        K = K_arr[k]
        payoff_sum = 0.0
        
        for p in range(N_paths):
            if exotic_type_id == 0:
                ST = paths[k, p, N_steps]
                if is_call[k]:
                    payoff_sum += max(ST - K, 0.0)
                else:
                    payoff_sum += max(K - ST, 0.0)

            elif exotic_type_id == 1:
                avg_S = 0.0
                for t in range(1, N_steps + 1):
                    avg_S += paths[k, p, t]
                avg_S /= N_steps
                
                if is_call[k]:
                    payoff_sum += max(avg_S - K, 0.0)
                else:
                    payoff_sum += max(K - avg_S, 0.0)
                    
            elif exotic_type_id == 2:
                extreme_S = paths[k, p, 1]
                if is_call[k]:
                    for t in range(2, N_steps + 1):
                        if paths[k, p, t] < extreme_S:
                            extreme_S = paths[k, p, t]
                    payoff_sum += max(paths[k, p, N_steps] - extreme_S, 0.0)
                else:
                    for t in range(2, N_steps + 1):
                        if paths[k, p, t] > extreme_S:
                            extreme_S = paths[k, p, t]
                    payoff_sum += max(extreme_S - paths[k, p, N_steps], 0.0)
                    
            elif exotic_type_id >= 3:
                hit = False
                for t in range(1, N_steps + 1):
                    if exotic_type_id == 3 and paths[k, p, t] >= barrier[k]:
                        hit = True; break
                    elif exotic_type_id == 4 and paths[k, p, t] >= barrier[k]:
                        hit = True; break
                    elif exotic_type_id == 5 and paths[k, p, t] <= barrier[k]:
                        hit = True; break
                    elif exotic_type_id == 6 and paths[k, p, t] <= barrier[k]:
                        hit = True; break
                
                if (exotic_type_id == 3 or exotic_type_id == 5) and not hit:
                    payoff_sum += rebate[k] * np.exp(r_arr[k] * T_arr[k])
                    continue
                if (exotic_type_id == 4 or exotic_type_id == 6) and hit:
                    payoff_sum += rebate[k] * np.exp(r_arr[k] * T_arr[k])
                    continue
                    
                ST = paths[k, p, N_steps]
                if is_call[k]:
                    payoff_sum += max(ST - K, 0.0)
                else:
                    payoff_sum += max(K - ST, 0.0)
                    
        result[k] = (payoff_sum / N_paths) * df
        
    return result

def _get_exotic_id(exotic_type: ExoticType) -> int:
    mapping = {
        ExoticType.VANILLA: 0,
        ExoticType.ASIAN: 1,
        ExoticType.LOOKBACK: 2,
        ExoticType.BARRIER_UI: 3,
        ExoticType.BARRIER_UO: 4,
        ExoticType.BARRIER_DI: 5,
        ExoticType.BARRIER_DO: 6
    }
    return mapping[exotic_type]

def price_exotic(contract: OptionContract, market: MarketState, N_paths: int = 16_384,
                 N_steps: int = 252, seed: int = None,
                 chunk_size: int = None) -> Union[float, np.ndarray]:
    """Price vanilla or exotic options via variance-reduced Monte Carlo.

    Supports Asian, lookback, and barrier payoffs with Sobol quasi-random
    sequences and BSM control variates. Optional chunked path processing
    to bound peak memory at scale.

    Args:
        contract: Option contract with exotic_type specification.
        market: Market state.
        N_paths: Number of simulation paths.
        N_steps: Number of time steps per path.
        seed: Random seed for Sobol sequence.
        chunk_size: If set, process paths in chunks of this size to reduce
            peak memory from O(N_paths) to O(chunk_size).

    Returns:
        Option price as float or ndarray.
    """
    b = np.broadcast(market.spot, contract.strike, contract.expiry, market.rate, market.volatility, market.dividend)

    S_arr = np.broadcast_to(market.spot, b.shape).astype(float).ravel()
    K_arr = np.broadcast_to(contract.strike, b.shape).astype(float).ravel()
    T_arr = np.broadcast_to(contract.expiry, b.shape).astype(float).ravel()
    r_arr = np.broadcast_to(market.rate, b.shape).astype(float).ravel()
    sig_arr = np.broadcast_to(market.volatility, b.shape).astype(float).ravel()
    q_arr = np.broadcast_to(market.dividend, b.shape).astype(float).ravel()

    bar_val = contract.barrier_level if contract.barrier_level is not None else 0.0
    barrier_arr = np.broadcast_to(bar_val, b.shape).astype(float).ravel()
    rebate_arr = np.broadcast_to(contract.rebate, b.shape).astype(float).ravel()

    is_call = np.full(b.shape, contract.option_type == OptionType.CALL, dtype=bool).ravel()
    exotic_id = _get_exotic_id(contract.exotic_type)

    if chunk_size is None or chunk_size >= N_paths:
        sampler = Sobol(d=N_steps, scramble=True, seed=seed)
        sobol_seq = sampler.random(N_paths)
        Z = norm.ppf(np.clip(sobol_seq, 1e-10, 1 - 1e-10))

        paths = _generate_paths_brownian_bridge(S_arr, r_arr, q_arr, sig_arr, T_arr, Z, N_steps, N_paths)
        result_arr = _price_exotic_numba(paths, K_arr, r_arr, T_arr, exotic_id, is_call, barrier_arr, rebate_arr)
    else:
        num_options = S_arr.shape[0]
        result_acc = np.zeros(num_options)
        paths_done = 0

        sampler = Sobol(d=N_steps, scramble=True, seed=seed)
        full_sobol = sampler.random(N_paths)

        while paths_done < N_paths:
            end = min(paths_done + chunk_size, N_paths)
            chunk_n = end - paths_done
            sobol_chunk = full_sobol[paths_done:end]
            Z_chunk = norm.ppf(np.clip(sobol_chunk, 1e-10, 1 - 1e-10))

            paths_chunk = _generate_paths_brownian_bridge(
                S_arr, r_arr, q_arr, sig_arr, T_arr, Z_chunk, N_steps, chunk_n)
            chunk_result = _price_exotic_numba(
                paths_chunk, K_arr, r_arr, T_arr, exotic_id, is_call, barrier_arr, rebate_arr)

            result_acc += chunk_result * chunk_n
            paths_done = end

        result_arr = result_acc / N_paths

    if contract.exotic_type == ExoticType.ASIAN or contract.exotic_type == ExoticType.LOOKBACK:
        if chunk_size is None or chunk_size >= N_paths:
            vanilla_mc = _price_exotic_numba(paths, K_arr, r_arr, T_arr, 0, is_call, barrier_arr, rebate_arr)
        else:
            vanilla_mc = result_arr

        vanilla_contract = OptionContract(
            strike=contract.strike,
            expiry=contract.expiry,
            option_type=contract.option_type,
            american=False,
            exotic_type=ExoticType.VANILLA
        )
        vanilla_bsm = black_scholes(vanilla_contract, market)
        if isinstance(vanilla_bsm, float):
            vanilla_bsm = np.array([vanilla_bsm])
        else:
            vanilla_bsm = vanilla_bsm.ravel()

        result_arr = result_arr - (vanilla_mc - vanilla_bsm)

    result = result_arr.reshape(b.shape)
    if result.ndim == 0:
        return float(result.item())
    return result


def monte_carlo_prices(contract: OptionContract, market: MarketState,
                       N: int = 32_768, seed: int = None,
                       chunk_size: int = None) -> Union[float, np.ndarray]:
    """Price a vanilla European option via variance-reduced Monte Carlo.

    Convenience wrapper around price_exotic with N_steps=1.

    Args:
        contract: Option contract.
        market: Market state.
        N: Number of MC paths.
        seed: Random seed.
        chunk_size: Optional chunk size for memory-bounded processing.

    Returns:
        Option price as float or ndarray.
    """
    return price_exotic(contract, market, N_paths=N, N_steps=1, seed=seed, chunk_size=chunk_size)

