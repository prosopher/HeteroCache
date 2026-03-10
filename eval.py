import argparse
import importlib

from common import EvalConfig, resolve_latest_checkpoint_for_alg


def load_eval_module(alg: str):
    module_name = f"{alg}.eval"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise SystemExit(f"Unsupported alg: {alg}") from exc
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("alg")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path or str(
        resolve_latest_checkpoint_for_alg(args.alg, output_root=args.output_root)
    )

    eval_config = EvalConfig(
        alg=args.alg,
        output_root=args.output_root,
        checkpoint_path=checkpoint_path,
    )

    eval_module = load_eval_module(args.alg)
    log_path = eval_module.run_eval(eval_config)

    print(f"Evaluation log: {log_path}")


if __name__ == "__main__":
    main()
