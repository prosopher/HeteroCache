import argparse
import importlib

from common import EvalConfig, add_dataclass_arguments, extract_dataclass_kwargs_from_namespace


def load_eval_module(alg: str):
    module_name = f"{alg}.eval"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise SystemExit(f"Unsupported alg: {alg}") from exc
        raise


def build_eval_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("alg")
    add_dataclass_arguments(
        parser,
        EvalConfig,
        exclude_fields={"alg"},
    )
    return parser


def main() -> None:
    parser = build_eval_parser()
    args = parser.parse_args()

    eval_config_kwargs = extract_dataclass_kwargs_from_namespace(
        EvalConfig,
        args,
        exclude_fields={"alg"},
    )
    eval_config = EvalConfig(
        alg=args.alg,
        **eval_config_kwargs,
    )

    eval_module = load_eval_module(args.alg)
    log_path = eval_module.run_eval(eval_config)

    print(f"Evaluation log: {log_path}")


if __name__ == "__main__":
    main()
