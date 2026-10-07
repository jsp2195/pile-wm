import argparse
from pile_wm.config import load_config, setup
from pile_wm.training import train_decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    cfg = load_config(parser.parse_args().config)
    train_decoder(cfg, setup(cfg))


if __name__ == "__main__":
    main()
