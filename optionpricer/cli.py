import argparse
import sys
from optionpricer.core import OptionContract, MarketState
from optionpricer import black_scholes, build_tree, monte_carlo_prices, crank_nicolson_fdm

def main():
    parser = argparse.ArgumentParser(
        prog="optionpricer",
        description="A high-performance quantitative option pricing CLI.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="model", help="Pricing model to use")

    parser_bs = subparsers.add_parser("bs", help="Black-Scholes analytical pricer")
    parser_bin = subparsers.add_parser("binomial", help="Cox-Ross-Rubinstein binomial tree")
    parser_mc = subparsers.add_parser("mc", help="Monte Carlo simulator")
    parser_fdm = subparsers.add_parser("fdm", help="Crank-Nicolson PSOR Finite Difference Method")

    for p in [parser_bs, parser_bin, parser_mc, parser_fdm]:
        p.add_argument("-S", type=float, required=True, help="Current asset price")
        p.add_argument("-K", type=float, required=True, help="Strike price")
        p.add_argument("-T", type=float, required=True, help="Time to maturity (years)")
        p.add_argument("-r", type=float, required=True, help="Risk free rate")
        p.add_argument("-q", type=float, default=0.0, help="Continuous dividend yield")
        p.add_argument("--sigma", type=float, required=True, help="Volatility")
        p.add_argument("--type", type=str, choices=["call", "put"], required=True, help="Option type")

    parser_bin.add_argument("-N", type=int, default=1000, help="Number of steps in binomial tree")
    parser_mc.add_argument("-N", type=int, default=32768, help="Number of Monte Carlo paths")
    parser_mc.add_argument("--seed", type=int, default=42, help="Random seed")
    parser_fdm.add_argument("-M", type=int, default=400, help="Number of spatial steps for FDM grid")
    parser_fdm.add_argument("-N", type=int, default=400, help="Number of time steps for FDM grid")

    args = parser.parse_args()
    if not args.model:
        parser.print_help()
        sys.exit(1)

    contract = OptionContract(strike=args.K, expiry=args.T, option_type=args.type)
    market = MarketState(spot=args.S, rate=args.r, volatility=args.sigma, dividend=args.q)

    if args.model == "bs":
        price = black_scholes(contract, market)
        print(f"Black-Scholes Price: {price:.4f}")
    elif args.model == "binomial":
        price = build_tree(contract, market, N=args.N)
        print(f"Binomial Tree Price: {price:.4f}")
    elif args.model == "mc":
        price = monte_carlo_prices(contract, market, N=args.N, seed=args.seed)
        print(f"Monte Carlo Price: {price:.4f}")
    elif args.model == "fdm":
        price = crank_nicolson_fdm(contract, market, M=args.M, N=args.N)
        print(f"Crank-Nicolson FDM Price: {price:.4f}")

if __name__ == "__main__":
    main()
