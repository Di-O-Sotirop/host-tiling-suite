from scripts.benchmark_runner import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file (e.g., config/gemver.yaml)")
    args = parser.parse_args()
    main(args.config)
