from dataclasses import dataclass
from typing import Union
import numpy as np

@dataclass
class OptionContract:
    strike: Union[float, np.ndarray]
    expiry: Union[float, np.ndarray]
    option_type: str = "call"
    american: bool = False

@dataclass
class MarketState:
    spot: Union[float, np.ndarray]
    rate: Union[float, np.ndarray]
    volatility: Union[float, np.ndarray]
    dividend: Union[float, np.ndarray] = 0.0
