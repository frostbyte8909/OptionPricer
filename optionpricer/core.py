from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from typing import Union, Optional, Any
import numpy as np


class ExoticType(str, Enum):
    VANILLA = "vanilla"
    ASIAN = "asian"
    LOOKBACK = "lookback"
    BARRIER_UI = "barrier_ui"
    BARRIER_UO = "barrier_uo"
    BARRIER_DI = "barrier_di"
    BARRIER_DO = "barrier_do"


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class OptionContract(BaseModel):
    """Specification for an option contract.

    Attributes:
        strike: Strike price(s), scalar or array for vectorized pricing.
        expiry: Time to expiry in years.
        option_type: 'call' or 'put'.
        american: Whether the option has early exercise rights.
        exotic_type: Type of exotic payoff (default: vanilla).
        barrier_level: Barrier level for barrier options.
        rebate: Rebate paid on barrier knock-out/knock-in.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strike: Union[float, np.ndarray]
    expiry: Union[float, np.ndarray]
    option_type: OptionType = OptionType.CALL
    american: bool = False
    exotic_type: ExoticType = ExoticType.VANILLA

    barrier_level: Optional[Union[float, np.ndarray]] = None
    rebate: Optional[Union[float, np.ndarray]] = 0.0

    @field_validator('strike', 'expiry', 'barrier_level', 'rebate', mode='before')
    def validate_numeric(cls, v):
        if v is None:
            return v
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            return float(arr.item())
        return arr


class MarketState(BaseModel):
    """Snapshot of market conditions for pricing.

    Attributes:
        spot: Current asset price(s), scalar or array.
        rate: Risk-free interest rate.
        volatility: Implied or realized volatility.
        dividend: Continuous dividend yield.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    spot: Union[float, np.ndarray]
    rate: Union[float, np.ndarray]
    volatility: Union[float, np.ndarray]
    dividend: Union[float, np.ndarray] = 0.0

    @field_validator('spot', 'rate', 'volatility', 'dividend', mode='before')
    def validate_numeric(cls, v):
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            return float(arr.item())
        return arr


def require_european(contract: OptionContract, model_name: str):
    """Guard that raises if contract has early exercise rights.

    Args:
        contract: Option contract to validate.
        model_name: Name of the model for the error message.

    Raises:
        ValueError: If contract.american is True.
    """
    if contract.american:
        raise ValueError(f"{model_name} does not support American-style exercise.")


def require_vanilla(contract: OptionContract, model_name: str):
    """Guard that raises if contract has exotic features.

    Args:
        contract: Option contract to validate.
        model_name: Name of the model for the error message.

    Raises:
        ValueError: If contract.exotic_type is not VANILLA.
    """
    if contract.exotic_type != ExoticType.VANILLA:
        raise ValueError(f"{model_name} only supports vanilla payoffs, got {contract.exotic_type.value}.")


def clamp_volatility(sigma: float, floor: float = 1e-4) -> float:
    """Enforce a volatility floor to prevent division-by-zero in d1.

    Args:
        sigma: Input volatility.
        floor: Minimum allowed volatility.

    Returns:
        max(sigma, floor).
    """
    return max(sigma, floor)
