import argparse
import importlib
from pathlib import Path


def load_train_module(alg: str):
    module_name = f"{alg}.train"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise SystemExit(f"Unsupported alg: {alg}") from exc
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("alg")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    train_module = load_train_module(args.alg)
    config = train_module.TrainConfig(
        alg=args.alg,
        output_root=args.output_root,
    )
    final_checkpoint = Path(train_module.run_train(config))

    print(f"Saved outputs to {final_checkpoint.parent}")
    print(f"Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
