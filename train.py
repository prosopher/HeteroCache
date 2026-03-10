import argparse
from pathlib import Path

from common import build_timestamped_output_dir
from heterocache.train import TrainConfig as HeteroCacheTrainConfig
from heterocache.train import CONFIG as HETEROCACHE_CONFIG
from heterocache.train import run_train as run_heterocache_train
from lsc.train import TrainConfig as LSCTrainConfig
from lsc.train import CONFIG as LSC_CONFIG
from lsc.train import run_train as run_lsc_train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("alg", choices=["heterocache", "lsc"])
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    output_dir = build_timestamped_output_dir(args.alg, output_root=args.output_root)

    if args.alg == "heterocache":
        config = HeteroCacheTrainConfig(**HETEROCACHE_CONFIG.__dict__)
        config.output_dir = str(output_dir)
        final_checkpoint = run_heterocache_train(config)
    else:
        config = LSCTrainConfig(**LSC_CONFIG.__dict__)
        config.output_dir = str(output_dir)
        final_checkpoint = run_lsc_train(config)

    print(f"Saved outputs to {Path(final_checkpoint).parent}")
    print(f"Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
