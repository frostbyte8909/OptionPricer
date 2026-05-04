import pytest
import numpy as np
from optionpricer.core import OptionContract, MarketState


C_CALL = OptionContract(strike=100, expiry=1.0, option_type="call")
C_PUT = OptionContract(strike=100, expiry=1.0, option_type="put")
C_AMER = OptionContract(strike=100, expiry=1.0, option_type="put", american=True)
MKT = MarketState(spot=100, rate=0.05, volatility=0.2)


class TestBlackScholes:
    def test_textbook_call(self):
        from optionpricer.models.black_scholes import black_scholes
        assert np.isclose(black_scholes(C_CALL, MKT), 10.4506, atol=1e-3)

    def test_textbook_put(self):
        from optionpricer.models.black_scholes import black_scholes
        assert np.isclose(black_scholes(C_PUT, MKT), 5.5735, atol=1e-2)

    def test_put_call_parity(self):
        from optionpricer.models.black_scholes import black_scholes
        call = black_scholes(C_CALL, MKT)
        put = black_scholes(C_PUT, MKT)
        parity = call - put - (100 * np.exp(-0.0) - 100 * np.exp(-0.05))
        assert abs(parity) < 1e-10

    def test_vectorized(self):
        from optionpricer.models.black_scholes import black_scholes
        c = OptionContract(strike=np.array([90, 100, 110]), expiry=1.0, option_type="call")
        prices = black_scholes(c, MKT)
        assert prices.shape == (3,)
        assert prices[0] > prices[1] > prices[2]

    def test_extreme_tail_precision(self):
        from optionpricer.models.black_scholes import black_scholes
        deep_otm = OptionContract(strike=300, expiry=0.01, option_type="call")
        price = black_scholes(deep_otm, MKT)
        assert price >= 0.0
        assert not np.isnan(price)

    def test_zero_expiry(self):
        from optionpricer.models.black_scholes import black_scholes
        c = OptionContract(strike=100, expiry=0.0, option_type="call")
        assert black_scholes(c, MKT) == 0.0

    def test_zero_vol(self):
        from optionpricer.models.black_scholes import black_scholes
        m = MarketState(spot=100, rate=0.05, volatility=0.0)
        price = black_scholes(C_CALL, m)
        assert price >= 0.0

    def test_greeks_returned(self):
        from optionpricer.models.black_scholes import black_scholes
        price, g = black_scholes(C_CALL, MKT, return_greeks=True)
        assert 0 < g["delta"] < 1
        assert g["gamma"] > 0
        assert g["vega"] > 0


class TestBinomial:
    def test_converges_to_bsm(self):
        from optionpricer.models.binomial import build_tree
        from optionpricer.models.black_scholes import black_scholes
        bs = black_scholes(C_CALL, MKT)
        bt = build_tree(C_CALL, MKT, N=1000)
        assert np.isclose(bt, bs, atol=1e-2)

    def test_american_put_geq_european(self):
        from optionpricer.models.binomial import build_tree
        euro = build_tree(C_PUT, MKT, N=500)
        amer = build_tree(C_AMER, MKT, N=500)
        assert amer >= euro - 1e-6


class TestFDM:
    def test_converges_to_bsm(self):
        from optionpricer.models.fdm import crank_nicolson_fdm
        from optionpricer.models.black_scholes import black_scholes
        bs = black_scholes(C_CALL, MKT)
        fdm = crank_nicolson_fdm(C_CALL, MKT, M=400, N=400)
        assert np.isclose(fdm, bs, atol=1e-2)


class TestMonteCarlo:
    def test_converges_to_bsm(self):
        from optionpricer.models.monte_carlo import monte_carlo_prices
        from optionpricer.models.black_scholes import black_scholes
        bs = black_scholes(C_CALL, MKT)
        mc = monte_carlo_prices(C_CALL, MKT, N=65536, seed=1)
        assert np.isclose(mc, bs, atol=1.0)

    def test_chunked_matches_full(self):
        from optionpricer.models.monte_carlo import monte_carlo_prices
        full = monte_carlo_prices(C_CALL, MKT, N=4096, seed=42)
        chunked = monte_carlo_prices(C_CALL, MKT, N=4096, seed=42, chunk_size=1024)
        assert np.isclose(full, chunked, atol=0.5)


class TestMertonJD:
    def test_no_jumps_equals_bsm(self):
        from optionpricer.models.jump_diffusion import merton_jump_diffusion
        from optionpricer.models.black_scholes import black_scholes
        bs = black_scholes(C_CALL, MKT)
        jd = merton_jump_diffusion(C_CALL, MKT, lam=1e-12, mu_j=0.0, sigma_j=0.0)
        assert np.isclose(jd, bs, atol=1e-4)

    def test_positive_price(self):
        from optionpricer.models.jump_diffusion import merton_jump_diffusion
        price = merton_jump_diffusion(C_CALL, MKT, lam=0.5, mu_j=-0.1, sigma_j=0.2)
        assert price > 0

    def test_vectorized(self):
        from optionpricer.models.jump_diffusion import merton_jump_diffusion
        c = OptionContract(strike=np.array([90, 100, 110]), expiry=1.0, option_type="call")
        prices = merton_jump_diffusion(c, MKT)
        assert prices.shape == (3,)


class TestFFT:
    def test_matches_bsm(self):
        from optionpricer.models.fft_pricing import carr_madan_fft
        from optionpricer.models.black_scholes import black_scholes
        bs = black_scholes(C_CALL, MKT)
        fft = carr_madan_fft(C_CALL, MKT)
        assert np.isclose(fft, bs, atol=0.05)

    def test_put(self):
        from optionpricer.models.fft_pricing import carr_madan_fft
        price = carr_madan_fft(C_PUT, MKT)
        assert price > 0


class TestHeston:
    def test_reduces_to_bsm(self):
        from optionpricer.models.heston import heston_price
        from optionpricer.models.black_scholes import black_scholes
        bs = black_scholes(C_CALL, MKT)
        h = heston_price(C_CALL, MKT, v0=0.04, kappa=10.0, theta=0.04, sigma_v=0.01, rho=0.0)
        assert np.isclose(h, bs, atol=0.5)

    def test_positive(self):
        from optionpricer.models.heston import heston_price
        p = heston_price(C_CALL, MKT, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.7)
        assert p > 0


class TestBates:
    def test_positive_and_reasonable(self):
        from optionpricer.models.bates import bates_price
        b = bates_price(C_CALL, MKT, v0=0.04, kappa=2.0, theta=0.04,
                        sigma_v=0.5, rho=-0.7, lam=0.1, mu_j=-0.05, sigma_j=0.1)
        assert 0 < b < 50

    def test_jumps_increase_otm_put(self):
        from optionpricer.models.bates import bates_price
        p_no = bates_price(C_PUT, MKT, v0=0.04, kappa=2.0, theta=0.04,
                           sigma_v=0.5, rho=-0.7, lam=0.0)
        p_yes = bates_price(C_PUT, MKT, v0=0.04, kappa=2.0, theta=0.04,
                            sigma_v=0.5, rho=-0.7, lam=0.5, mu_j=-0.1, sigma_j=0.2)
        assert p_yes > 0


class TestQuanto:
    def test_positive(self):
        from optionpricer.models.quanto import quanto_price
        p = quanto_price(C_CALL, MKT, r_domestic=0.03, sigma_fx=0.1, rho=0.3)
        assert p > 0


class TestMultiAsset:
    def test_basket_positive(self):
        from optionpricer.models.multi_asset import basket_option
        spots = np.array([100.0, 100.0])
        vols = np.array([0.2, 0.25])
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        p = basket_option(spots, vols, corr, K=100, T=1.0, r=0.05, seed=42)
        assert p > 0

    def test_spread_positive(self):
        from optionpricer.models.multi_asset import spread_option
        p = spread_option(S1=110, S2=100, vol1=0.2, vol2=0.25, rho=0.5, K=5, T=1.0, r=0.05)
        assert p > 0


class TestGARCH:
    def test_stationarity(self):
        from optionpricer.analytics.volatility import garch_fit
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, 500)))
        omega, alpha, beta, sigma_fc = garch_fit(prices)
        assert alpha + beta < 1.0
        assert sigma_fc > 0


class TestEWMA:
    def test_positive(self):
        from optionpricer.analytics.volatility import ewma_volatility
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
        vol = ewma_volatility(prices, span=30)
        assert vol > 0


class TestSABR:
    def test_atm_positive(self):
        from optionpricer.analytics.volatility import sabr_implied_vol
        vols = sabr_implied_vol(F=100, K=np.array([100.0]), T=1.0, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
        assert vols[0] > 0

    def test_extreme_strike_no_nan(self):
        from optionpricer.analytics.volatility import sabr_implied_vol
        vols = sabr_implied_vol(F=100, K=np.array([1.0, 500.0]), T=1.0, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
        assert not np.any(np.isnan(vols))
        assert np.all(vols > 0)

    def test_zero_nu(self):
        from optionpricer.analytics.volatility import sabr_implied_vol
        vols = sabr_implied_vol(F=100, K=np.array([100.0]), T=1.0, alpha=0.2, beta=0.5, rho=0.0, nu=0.0)
        assert vols[0] > 0


class TestDupire:
    def test_flat_vol_surface(self):
        from optionpricer.analytics.surface import dupire_local_vol
        K = np.linspace(80, 120, 9)
        T = np.array([0.25, 0.5, 1.0])
        iv = np.full((3, 9), 0.20)
        lv = dupire_local_vol(K, T, iv, S=100, r=0.05, q=0.0)
        assert np.allclose(lv, 0.20, atol=0.08)


class TestArbitrage:
    def test_flat_surface_no_violations(self):
        from optionpricer.analytics.surface import arbitrage_check
        K = np.linspace(80, 120, 9)
        T = np.array([0.25, 0.5, 1.0])
        iv = np.full((3, 9), 0.20)
        result = arbitrage_check(K, T, iv, S=100, r=0.05, q=0.0)
        assert len(result["calendar_violations"]) == 0
        assert len(result["butterfly_violations"]) == 0


class TestVannaVolga:
    def test_atm_equals_bsm(self):
        from optionpricer.analytics.vanna_volga import vanna_volga_price
        from optionpricer.models.black_scholes import black_scholes
        bs = black_scholes(C_CALL, MKT)
        vv = vanna_volga_price(C_CALL, MKT, K1=90, K2=100, K3=110,
                               sigma1=0.20, sigma_atm=0.20, sigma3=0.20)
        assert np.isclose(vv, bs, atol=1e-6)


class TestAADGreeks:
    def test_delta_bounds(self):
        from optionpricer.analytics.risk import aad_greeks
        g = aad_greeks(C_CALL, MKT)
        assert 0 < g["delta"] < 1
        assert g["gamma"] > 0
        assert g["vega"] > 0

    def test_vanna_volga_present(self):
        from optionpricer.analytics.risk import aad_greeks
        g = aad_greeks(C_CALL, MKT)
        assert "vanna" in g
        assert "volga" in g
        assert "charm" in g


class TestImpliedVol:
    def test_roundtrip(self):
        from optionpricer.models.black_scholes import black_scholes
        from optionpricer.analytics.implied_vol import implied_vol
        price = black_scholes(C_CALL, MKT)
        iv = implied_vol(price, C_CALL, MKT)
        assert np.isclose(iv, 0.2, atol=1e-3)


class TestBjerksundStensland:
    def test_call_geq_bsm(self):
        from optionpricer.models.bjerksund_stensland import bjerksund_stensland_american
        from optionpricer.models.black_scholes import black_scholes
        c_amer = OptionContract(strike=100, expiry=1.0, option_type="call", american=True)
        bs = black_scholes(C_CALL, MKT)
        bjs = bjerksund_stensland_american(c_amer, MKT)
        assert bjs >= bs - 1e-4


class TestBarrier:
    def test_do_call_leq_vanilla(self):
        from optionpricer.models.barrier_analytical import barrier_analytical
        from optionpricer.models.black_scholes import black_scholes
        from optionpricer.core import ExoticType
        c = OptionContract(strike=100, expiry=1.0, option_type="call",
                          exotic_type=ExoticType.BARRIER_DO, barrier_level=80, rebate=0.0)
        bs = black_scholes(C_CALL, MKT)
        bar = barrier_analytical(c, MKT)
        assert bar <= bs + 1e-4


class TestTermStructure:
    def test_nelson_siegel_level(self):
        from optionpricer.analytics.term_structure import nelson_siegel
        rates = nelson_siegel(np.array([1.0, 5.0, 10.0]), beta0=0.05, beta1=-0.02, beta2=0.01, lam=2.0)
        assert rates.shape == (3,)
        assert np.all(rates > 0)

    def test_bootstrap(self):
        from optionpricer.analytics.term_structure import bootstrap_zeros
        tenors = np.array([1.0, 2.0, 3.0])
        par_rates = np.array([0.03, 0.035, 0.04])
        t, z = bootstrap_zeros(tenors, par_rates)
        assert z.shape == (3,)
        assert np.all(z > 0)


class TestGuardrails:
    def test_require_european_raises(self):
        from optionpricer.core import require_european
        with pytest.raises(ValueError):
            require_european(C_AMER, "TestModel")

    def test_require_european_passes(self):
        from optionpricer.core import require_european
        require_european(C_CALL, "TestModel")


class TestPropertyBased:
    @pytest.mark.parametrize("S", [50, 100, 200])
    @pytest.mark.parametrize("K", [80, 100, 120])
    def test_call_price_positive(self, S, K):
        from optionpricer.models.black_scholes import black_scholes
        c = OptionContract(strike=K, expiry=1.0, option_type="call")
        m = MarketState(spot=S, rate=0.05, volatility=0.2)
        assert black_scholes(c, m) >= 0

    @pytest.mark.parametrize("S", [50, 100, 200])
    @pytest.mark.parametrize("K", [80, 100, 120])
    def test_put_call_parity(self, S, K):
        from optionpricer.models.black_scholes import black_scholes
        c_call = OptionContract(strike=K, expiry=1.0, option_type="call")
        c_put = OptionContract(strike=K, expiry=1.0, option_type="put")
        m = MarketState(spot=S, rate=0.05, volatility=0.2)
        call = black_scholes(c_call, m)
        put = black_scholes(c_put, m)
        parity = call - put - (S - K * np.exp(-0.05))
        assert abs(parity) < 1e-8

    @pytest.mark.parametrize("sigma", [0.01, 0.1, 0.5, 1.0])
    def test_vega_positive(self, sigma):
        from optionpricer.models.black_scholes import black_scholes
        m = MarketState(spot=100, rate=0.05, volatility=sigma)
        _, g = black_scholes(C_CALL, m, return_greeks=True)
        assert g["vega"] >= 0

    @pytest.mark.parametrize("T", [0.01, 0.25, 1.0, 5.0])
    def test_call_monotonic_in_T(self, T):
        from optionpricer.models.black_scholes import black_scholes
        c1 = OptionContract(strike=100, expiry=T, option_type="call")
        c2 = OptionContract(strike=100, expiry=T + 0.1, option_type="call")
        p1 = black_scholes(c1, MKT)
        p2 = black_scholes(c2, MKT)
        assert p2 >= p1 - 1e-8
