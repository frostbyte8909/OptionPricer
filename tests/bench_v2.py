"""
Benchmark suite for OptionPricer V2.

Measures wall-clock latency, peak memory, CPU utilization, and memory
bandwidth across all pricing kernels. Outputs a structured report.
"""
import time
import tracemalloc
import os
import sys
import json
import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optionpricer.core import OptionContract, MarketState, OptionType
from optionpricer.models.black_scholes import black_scholes
from optionpricer.models.binomial import build_tree
from optionpricer.models.fdm import crank_nicolson_fdm
from optionpricer.models.monte_carlo import monte_carlo_prices
from optionpricer.models.jump_diffusion import merton_jump_diffusion
from optionpricer.models.fft_pricing import carr_madan_fft
from optionpricer.analytics.risk import aad_greeks
from optionpricer.analytics.volatility import garch_fit, sabr_implied_vol, ewma_volatility
from optionpricer.analytics.implied_vol import implied_vol
from optionpricer.analytics.greeks import greeks


WARMUP_ITERS = 3
BENCH_ITERS = 50
PROCESS = psutil.Process(os.getpid())


def _cpu_times():
    t = PROCESS.cpu_times()
    return t.user + t.system


def bench_one(fn, label, iters=BENCH_ITERS, warmup=WARMUP_ITERS):
    for _ in range(warmup):
        fn()

    tracemalloc.start()
    mem_before = PROCESS.memory_info().rss
    cpu_before = _cpu_times()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)

    cpu_after = _cpu_times()
    mem_after = PROCESS.memory_info().rss
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall_ms = np.median(times)
    p99_ms = np.percentile(times, 99)
    cpu_s = cpu_after - cpu_before
    wall_total_s = sum(times) / 1e3
    cpu_ratio = cpu_s / wall_total_s if wall_total_s > 0 else 0.0
    rss_delta_kb = (mem_after - mem_before) / 1024
    peak_kb = peak_traced / 1024

    return {
        "label": label,
        "median_ms": round(wall_ms, 4),
        "p99_ms": round(p99_ms, 4),
        "cpu_ratio": round(cpu_ratio, 2),
        "rss_delta_kb": round(rss_delta_kb, 1),
        "peak_alloc_kb": round(peak_kb, 1),
    }


def run_suite():
    c_call = OptionContract(strike=100, expiry=1.0, option_type="call")
    c_put = OptionContract(strike=100, expiry=1.0, option_type="put", american=True)
    m = MarketState(spot=100, rate=0.05, volatility=0.2, dividend=0.0)

    vec_strikes = np.linspace(80, 120, 500)
    c_vec = OptionContract(strike=vec_strikes, expiry=1.0, option_type="call")

    np.random.seed(42)
    garch_prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, 500)))

    results = []

    results.append(bench_one(lambda: black_scholes(c_call, m), "BSM scalar"))
    results.append(bench_one(lambda: black_scholes(c_vec, m), "BSM vectorized (500)"))
    results.append(bench_one(lambda: black_scholes(c_call, m, return_greeks=True), "BSM + Greeks"))
    results.append(bench_one(lambda: build_tree(c_call, m, N=1000), "Binomial N=1000"))
    results.append(bench_one(lambda: build_tree(c_put, m, N=1000), "Binomial Amer N=1000"))
    results.append(bench_one(lambda: build_tree(c_put, m, N=5000), "Binomial Amer N=5000", iters=10))
    results.append(bench_one(lambda: crank_nicolson_fdm(c_call, m, M=200, N=200), "FDM 200x200"))
    results.append(bench_one(lambda: crank_nicolson_fdm(c_put, m, M=400, N=400), "FDM Amer 400x400", iters=10))
    results.append(bench_one(lambda: monte_carlo_prices(c_call, m, N=16384, seed=42), "MC 16K paths"))
    results.append(bench_one(lambda: merton_jump_diffusion(c_call, m, lam=0.1, mu_j=-0.05, sigma_j=0.1), "Merton JD"))
    results.append(bench_one(lambda: carr_madan_fft(c_call, m, N=4096), "FFT 4096"))
    results.append(bench_one(lambda: aad_greeks(c_call, m), "AAD Greeks"))
    results.append(bench_one(lambda: greeks(c_call, m, N=100), "FD Greeks Euro"))
    results.append(bench_one(lambda: greeks(c_put, m, N=100), "FD Greeks Amer"))
    results.append(bench_one(lambda: implied_vol(10.45, c_call, m), "IV Solve"))
    results.append(bench_one(lambda: garch_fit(garch_prices), "GARCH fit (500d)"))
    results.append(bench_one(lambda: ewma_volatility(garch_prices, span=30), "EWMA vol"))
    results.append(bench_one(
        lambda: sabr_implied_vol(100, np.linspace(80, 120, 100), 1.0, 0.2, 0.5, -0.3, 0.4),
        "SABR smile (100K)"))

    return results


def print_report(results):
    sep = "-" * 100
    print(f"\n{'='*100}")
    print(f"  OPTIONPRICER V2 — INSTITUTIONAL BENCHMARK REPORT")
    print(f"  Python {sys.version.split()[0]} | NumPy {np.__version__} | "
          f"CPU cores: {psutil.cpu_count(logical=False)}P/{psutil.cpu_count()}L | "
          f"RSS baseline: {PROCESS.memory_info().rss / 1024 / 1024:.1f} MB")
    print(f"{'='*100}\n")

    header = f"{'Kernel':<28} {'Median(ms)':>10} {'P99(ms)':>10} {'CPU/Wall':>10} {'RSS Δ(KB)':>10} {'Peak(KB)':>10}"
    print(header)
    print(sep)

    for r in results:
        cpu_bar = "█" * min(int(r["cpu_ratio"] * 5), 20)
        print(f"{r['label']:<28} {r['median_ms']:>10.4f} {r['p99_ms']:>10.4f} "
              f"{r['cpu_ratio']:>8.2f}x  {r['rss_delta_kb']:>9.1f} {r['peak_alloc_kb']:>9.1f}  {cpu_bar}")

    print(sep)

    fastest = min(results, key=lambda x: x["median_ms"])
    slowest = max(results, key=lambda x: x["median_ms"])
    heaviest = max(results, key=lambda x: x["peak_alloc_kb"])
    most_parallel = max(results, key=lambda x: x["cpu_ratio"])

    print(f"\n  Fastest kernel:      {fastest['label']} ({fastest['median_ms']:.4f} ms)")
    print(f"  Slowest kernel:      {slowest['label']} ({slowest['median_ms']:.4f} ms)")
    print(f"  Heaviest allocation: {heaviest['label']} ({heaviest['peak_alloc_kb']:.1f} KB peak)")
    print(f"  Best parallelism:    {most_parallel['label']} ({most_parallel['cpu_ratio']:.2f}x CPU/Wall)")
    print()


if __name__ == "__main__":
    results = run_suite()
    print_report(results)
