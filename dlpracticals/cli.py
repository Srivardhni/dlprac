"""Command-line launcher."""
import argparse
from importlib import import_module

def main():
    parser = argparse.ArgumentParser(description="Run a simplified DL practical")
    parser.add_argument("experiment", type=int, choices=range(1, 11))
    parser.add_argument("--reference", action="store_true", help="Use reference module")
    args = parser.parse_args()
    section = "reference" if args.reference else "simplified"
    module = import_module(f"dlpracticals.{section}.exp{args.experiment}")
    module.run()

if __name__ == "__main__":
    main()
