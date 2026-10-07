import argparse
from pile_wm.config import load_config, setup
from pile_wm.planning import run_planning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    cfg = load_config(parser.parse_args().config)
    run_planning(cfg, setup(cfg))


if __name__ == "__main__":
    main()
