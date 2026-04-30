import argparse

from optionpricer import black_scholes, build_tree, monte_carlo_prices

def main():
    """This is the entry point called by the terminal."""
    
    parser = argparse.ArgumentParser(
        prog="optionpricer",
        description="A high-performance quantitative option pricing CLI."
    )

    parser.add_argument("--model", type=str, choices=["bs", "binomial", "mc"], required=True, help="Pricing model to use")
    parser.add_argument("-S", type=float, required=True, help="Current asset price")
    parser.add_argument("-K", type=float, required=True, help="Strike price")
    parser.add_argument("-T", type=float, required=True, help="Time to maturity (years)")
    parser.add_argument("-r", type=float, required=True, help="Risk free rate")
    parser.add_argument("--sigma", type=float, required=True, help="Volatility")
    parser.add_argument("--type", type=str, choices=["call", "put"], required=True, help="Option type")
    
    args = parser.parse_args()

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
    # test using `python cli.py --help`
    main()
