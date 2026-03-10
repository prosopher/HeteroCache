import argparse

from common import EvalConfig, resolve_latest_checkpoint_for_alg
from heterocache.eval import run_eval as run_heterocache_eval
from lsc.eval import run_eval as run_lsc_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("alg", choices=["heterocache", "lsc"])
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path or str(
        resolve_latest_checkpoint_for_alg(args.alg, output_root=args.output_root)
    )

    eval_config = EvalConfig(checkpoint_path=checkpoint_path)
    if args.device is not None:
        eval_config.device = args.device

    if args.alg == "heterocache":
        log_path = run_heterocache_eval(eval_config)
    else:
        log_path = run_lsc_eval(eval_config)

    print(f"Evaluation log: {log_path}")


if __name__ == "__main__":
    main()
