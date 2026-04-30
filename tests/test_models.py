import pytest
import numpy as np
from optionpricer.core import OptionContract, MarketState
from optionpricer.models.black_scholes import black_scholes
from optionpricer.models.binomial import build_tree
from optionpricer.models.fdm import crank_nicolson_fdm

def test_black_scholes_textbook():
    contract = OptionContract(strike=100, expiry=1.0, option_type="call")
    market = MarketState(spot=100, rate=0.05, volatility=0.2, dividend=0.0)
    price = black_scholes(contract, market)
    assert np.isclose(price, 10.4506, atol=1e-4)

def test_binomial_convergence():
    contract = OptionContract(strike=100, expiry=1.0, option_type="call")
    market = MarketState(spot=100, rate=0.05, volatility=0.2)
    bs_price = black_scholes(contract, market)
    bin_price = build_tree(contract, market, N=1000)
    assert np.isclose(bin_price, bs_price, atol=1e-2)

def test_vectorized_black_scholes():
    contract = OptionContract(strike=np.array([90, 100, 110]), expiry=1.0, option_type="call")
    market = MarketState(spot=100, rate=0.05, volatility=0.2)
    prices = black_scholes(contract, market)
    assert prices.shape == (3,)
    assert prices[0] > prices[1] > prices[2]

def test_fdm_convergence():
    contract = OptionContract(strike=100, expiry=1.0, option_type="call")
    market = MarketState(spot=100, rate=0.05, volatility=0.2)
    bs_price = black_scholes(contract, market)
    fdm_price = crank_nicolson_fdm(contract, market, M=400, N=400)
    assert np.isclose(fdm_price, bs_price, atol=1e-2)
