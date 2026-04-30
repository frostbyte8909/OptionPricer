import argparse
import sys

from optionpricer import black_scholes, build_tree, monte_carlo_prices

def main():
    """This is the entry point called by the terminal."""
    
    parser = argparse.ArgumentParser(
        prog="optionpricer",
        description="A high-performance quantitative option pricing CLI.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="model", help="Pricing model to use")

    parser_bs = subparsers.add_parser(
        "bs", 
        help="Black-Scholes analytical pricer", 
        description=black_scholes.__doc__, 
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser_bin = subparsers.add_parser(
        "binomial", 
        help="Cox-Ross-Rubinstein binomial tree", 
        description=build_tree.__doc__, 
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser_mc = subparsers.add_parser(
        "mc", 
        help="Monte Carlo simulator", 
        description=monte_carlo_prices.__doc__, 
        formatter_class=argparse.RawTextHelpFormatter
    )

    for p in [parser_bs, parser_bin, parser_mc]:
        p.add_argument("-S", type=float, required=True, help="Current asset price")
        p.add_argument("-K", type=float, required=True, help="Strike price")
        p.add_argument("-T", type=float, required=True, help="Time to maturity (years)")
        p.add_argument("-r", type=float, required=True, help="Risk free rate")
        p.add_argument("--sigma", type=float, required=True, help="Volatility")
        p.add_argument("--type", type=str, choices=["call", "put"], required=True, help="Option type")

    args = parser.parse_args()

    if not args.model:
        parser.print_help()
        sys.exit(1)

    if args.model == "bs":
        price = black_scholes(args.S, args.K, args.T, args.r, args.sigma, option_type=args.type)
        print(f"Black-Scholes Price: {price:.4f}")
    elif args.model == "binomial":
        price = build_tree(args.S, args.K, args.T, args.r, args.sigma, option_type=args.type)
        print(f"Binomial Tree Price: {price:.4f}")
    elif args.model == "mc":
        price = monte_carlo_prices(args.S, args.K, args.T, args.r, args.sigma, option_type=args.type)
        print(f"Monte Carlo Price: {price:.4f}")
    

if __name__ == "__main__":
    main()
