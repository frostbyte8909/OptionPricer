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
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    spot: Union[float, np.ndarray]
    rate: Union[float, np.ndarray]
    volatility: Union[float, np.ndarray]
    dividend: Union[float, np.ndarray] = 0.0

    @field_validator('spot', 'rate', 'volatility', 'dividend', mode='before')
    def validate_numeric(cls, v):
        arr = np.asarray(v, dtype=np.float64)
        if np.any(arr < 0) and v != 'rate' and v != 'dividend':
            raise ValueError("Spot and Volatility must be non-negative.")
        if arr.ndim == 0:
            return float(arr.item())
        return arr
