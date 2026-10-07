import argparse
from pile_wm.config import load_config
from pile_wm.reporting import create_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    create_report(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
