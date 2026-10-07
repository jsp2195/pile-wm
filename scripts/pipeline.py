import argparse
from pile_wm.config import load_config
from pile_wm.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--regenerate", action="store_true", help="regenerate state data even when its manifest matches")
    args = parser.parse_args()
    run_pipeline(load_config(args.config), args.regenerate)


if __name__ == "__main__":
    main()
