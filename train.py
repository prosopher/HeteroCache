import argparse
import importlib
from pathlib import Path

from common import add_dataclass_arguments, build_dataclass_kwargs_from_json_and_namespace


def load_train_module(alg: str):
    module_name = f"{alg}.train"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise SystemExit(f"Unsupported alg: {alg}") from exc
        raise


def build_train_parser(alg: str):
    train_module = load_train_module(alg)

    parser = argparse.ArgumentParser()
    parser.add_argument("alg")
    parser.add_argument(
        "--default-config-path",
        dest="default_config_path",
        default=f"configs/train_{alg}.json",
    )
    add_dataclass_arguments(
        parser,
        train_module.TrainConfig,
        exclude_fields={"alg"},
    )
    return parser, train_module


def main() -> None:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("alg")
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    parser, train_module = build_train_parser(bootstrap_args.alg)
    args = parser.parse_args()

    config_kwargs = build_dataclass_kwargs_from_json_and_namespace(
        config_cls=train_module.TrainConfig,
        default_config_path=args.default_config_path,
        args=args,
        exclude_fields={"alg"},
    )
    config = train_module.TrainConfig(
        alg=args.alg,
        **config_kwargs,
    )

    final_checkpoint = Path(train_module.run_train(config))

    print(f"Saved outputs to {final_checkpoint.parent}")
    print(f"Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
