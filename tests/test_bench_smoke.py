"""Smoke tests for the benchmark harness (no absolute latency thresholds)."""

import math

import pytest

from optionpricer.core import OptionContract, MarketState
from optionpricer.models.black_scholes import black_scholes
from optionpricer.models.binomial import build_tree
from tests.bench_v2 import bench_one


@pytest.mark.bench_smoke
def test_bench_one_bsm_scalar_finite():
    c = OptionContract(strike=100, expiry=1.0, option_type="call")
    m = MarketState(spot=100, rate=0.05, volatility=0.2, dividend=0.0)
    r = bench_one(lambda: black_scholes(c, m), "smoke_bsm", iters=1, warmup=0)
    assert math.isfinite(r["median_ms"])
    assert r["median_ms"] >= 0.0


@pytest.mark.bench_smoke
def test_bench_one_binomial_finite():
    c = OptionContract(strike=100, expiry=1.0, option_type="call")
    m = MarketState(spot=100, rate=0.05, volatility=0.2, dividend=0.0)
    r = bench_one(lambda: build_tree(c, m, N=32), "smoke_binomial", iters=1, warmup=0)
    assert math.isfinite(r["median_ms"])
    assert r["median_ms"] >= 0.0
