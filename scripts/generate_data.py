import argparse
import json
from pile_wm.config import load_config, setup
from pile_wm.data import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    metadata = generate(config, setup(config))
    print(json.dumps({"generation_seconds": metadata["generation_seconds"], "splits": {k: len(v) for k, v in metadata["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
